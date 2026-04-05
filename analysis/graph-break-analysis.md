======================================================================
EXPLAIN ANALYSIS — Graph Break Deep Dive
======================================================================

Models analyzed: 106
Results (model × mode): 208 ok, 6 errors

Total graph breaks across all models: 1254
Average breaks per model-mode: 6.0
Max breaks in a single model-mode: 30
Average subgraphs per model-mode: 7.0

──────────────────────────────────────────────────
GRAPH BREAK COUNT DISTRIBUTION
──────────────────────────────────────────────────

      Breaks   Count       %
0 (full_graph)      18    8.7%  ████
           1       6    2.9%  █
         2-3      72   34.6%  █████████████████
         4-5      24   11.5%  █████
        6-10      55   26.4%  █████████████
         11+      33   15.9%  ███████

──────────────────────────────────────────────────
TOP MODELS BY GRAPH BREAK COUNT
──────────────────────────────────────────────────

Model                                      eval  train  total  graphs
EncodecModel                                 29     30     59      61
AriaModel                                    27     27     54      56
VitsModel                                    22     29     51      53
MimiModel                                    24     24     48      50
PPDocLayoutV3Model                            9     21     30      32
AriaTextModel                                14     14     28      30
ReformerModel                                14     13     27      29
DFineModel                                    8     18     26      28
SwitchTransformersModel                      13     13     26      28
GroundingDinoModel                           12     12     24      26
LongcatFlashModel                            12     12     24      26
MMGroundingDinoModel                         12     12     24      26
Glm4vMoeVisionModel                          11     11     22      24
Glm4vVisionModel                             11     11     22      24
DeformableDetrModel                          10     10     20      22
GraniteMoeHybridModel                        10     10     20      22
PeAudioModel                                 10     10     20      22
IBertModel                                    0     19     19      21
PPDocLayoutV2Model                            5     14     19      21
RTDetrModel                                   5     14     19      21
  ... and 86 more models

──────────────────────────────────────────────────
ROOT CAUSE TAXONOMY (all breaks, not just first)
──────────────────────────────────────────────────

  #  Root Cause                           Breaks  Models      %
  1  Other                                  1319     103  81.6%
  2  Data-dependent branching                155      69   9.6%
       ↳ Control flow depends on tensor values, not shapes. e.g. if tensor.sum() > 0, or aten._local_scalar_dense
  3  copy.deepcopy()                          53      27   3.3%
  4  Tensor.item()                            25       8   1.5%
  5  Dynamic shape operator                   22       6   1.4%
       ↳ Op output shape depends on input data, not just input shapes. e.g. aten.nonzero (data-dependent output size), aten.repeat_interleave
  6  Unsupported method/builtin               20       8   1.2%
       ↳ Dynamo can't trace a specific method or builtin. e.g. ContiguousFormat.get(), RNG .seed(), context manager 'lock'
  7  logging.Logger                           14       7   0.9%
  8  as_proxy() missing                        6       3   0.4%
       ↳ Dynamo can't convert certain arg types to proxy during tracing. e.g. DETR models pass ValueError/bool to functions Dynamo can't represent
  9  Tensor requires_grad mutation             3       2   0.2%

     Total                                  1617

──────────────────────────────────────────────────
ACTIONABILITY BREAKDOWN
──────────────────────────────────────────────────

Level                         Breaks  Models      %
────────────────────────────────────────────────────
Fixable in user code              81      36   5.0%
  └ copy.deepcopy()                     53  Replace deepcopy with clone() or manual copy
  └ Tensor.item()                       25  Remove .item() calls or guard with torch.compiler.is_compiling()
  └ Tensor requires_grad mutation        3  Restructure gradient handling to avoid in-place mutation
Needs library PR                  40      16   2.5%
  └ Unsupported method/builtin          20  Add Dynamo tracing for unsupported builtins
  └ logging.Logger                      14  Dynamo should trace through logging calls — PyTorch fix
  └ as_proxy() missing                   6  Add proxy support for additional arg types in Dynamo
Needs compiler change            177      75  10.9%
  └ Data-dependent branching           155  Requires torch.cond() or model restructuring — no quick fix
  └ Dynamic shape operator              22  Op output shape depends on data — fundamental limitation
Needs investigation             1319     103  81.6%
  └ Other                             1319  Requires manual investigation
────────────────────────────────────────────────────
Total                           1617

──────────────────────────────────────────────────
TOP 15 RAW BREAK REASONS
──────────────────────────────────────────────────

  1. [900x] Graph break (user stack suppressed due to duplicate graph break) in user code at /data/users/pengwu/

  2. [447x] Graph break in user code at /data/users/pengwu/dotsync-home/envs/graph-break-corpus/lib/python3.12/s

  3. [250x] Graph break: torch.compile cannot properly resume from this graph break, which results in a skip.
to

  4. [16x] Graph break from `Tensor.item()`, consider setting:
    torch._dynamo.config.capture_scalar_outputs 

  5. [2x] torch._dynamo hit config.recompile_limit (8)
   function: 'torch_dynamo_resume_in_forward_at_162' (/

  6. [2x] torch._dynamo hit config.recompile_limit (8)
   function: 'torch_dynamo_resume_in_forward_at_343' (/
