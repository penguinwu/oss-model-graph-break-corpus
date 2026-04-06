======================================================================
EXPLAIN ANALYSIS — Graph Break Deep Dive
======================================================================

Models analyzed: 105
Results (model × mode): 208 ok, 6 errors

Total graph breaks across all models: 1230
Average breaks per model-mode: 5.9
Max breaks in a single model-mode: 30
Average subgraphs per model-mode: 6.9

──────────────────────────────────────────────────
GRAPH BREAK COUNT DISTRIBUTION
──────────────────────────────────────────────────

      Breaks   Count       %
0 (full_graph)      20    9.6%  ████
           1       6    2.9%  █
         2-3      72   34.6%  █████████████████
         4-5      24   11.5%  █████
        6-10      54   26.0%  ████████████
         11+      32   15.4%  ███████

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
MMGroundingDinoModel                         12     12     24      26
Glm4vMoeVisionModel                          11     11     22      24
Glm4vVisionModel                             11     11     22      24
LongcatFlashModel                            11     11     22      24
DeformableDetrModel                          10     10     20      22
GraniteMoeHybridModel                        10     10     20      22
PeAudioModel                                 10     10     20      22
IBertModel                                    0     19     19      21
PPDocLayoutV2Model                            5     14     19      21
RTDetrModel                                   5     14     19      21
  ... and 85 more models

──────────────────────────────────────────────────
ROOT CAUSE TAXONOMY (all breaks, not just first)
──────────────────────────────────────────────────

  #  Root Cause                           Breaks  Models      %
  1  Data-dependent branching                537      81  33.6%
       ↳ Control flow depends on tensor values, not shapes. e.g. if tensor.sum() > 0, or aten._local_scalar_dense
  2  Other                                   216      61  13.5%
  3  copy.deepcopy()                         205      26  12.8%
  4  as_proxy() missing                      150      10   9.4%
       ↳ Dynamo can't convert certain arg types to proxy during tracing. e.g. DETR models pass ValueError/bool to functions Dynamo can't represent
  5  Dynamic shape operator                  122      16   7.6%
       ↳ Op output shape depends on input data, not just input shapes. e.g. aten.nonzero (data-dependent output size), aten.repeat_interleave
  6  Skipped function call                   100      16   6.3%
       ↳ Dynamo devs marked a function as not-traceable. e.g. audio models call importlib.util.find_spec during forward()
  7  Tensor.item()                            99      18   6.2%
  8  Unsupported method/builtin               69       8   4.3%
       ↳ Dynamo can't trace a specific method or builtin. e.g. ContiguousFormat.get(), RNG .seed(), context manager 'lock'
  9  logging.Logger                           49       9   3.1%
 10  Tensor requires_grad mutation            40      12   2.5%
 11  Non-Tensor return                        10       1   0.6%
       ↳ A torch op returns a non-Tensor value Dynamo can't trace. e.g. torch.* ops returning ints or tuples of non-Tensors

     Total                                  1597

──────────────────────────────────────────────────
TOP 15 RAW BREAK REASONS
──────────────────────────────────────────────────

  1. [491x] Graph break (user stack suppressed due to duplicate graph break) in user code at transformers/models

  2. [246x] Graph break: torch.compile cannot properly resume from this graph break, which results in a skip.
to

  3. [219x] Graph break (user stack suppressed due to duplicate graph break) in user code at transformers/utils/

  4. [117x] Graph break (user stack suppressed due to duplicate graph break) in user code at transformers/config

  5. [52x] Graph break in user code at transformers/configuration_utils.py:1231
Graph Break Reason: Encountered

  6. [44x] Graph break (user stack suppressed due to duplicate graph break) in user code at torch/nn/functional

  7. [28x] Graph break in user code at transformers/utils/import_utils.py:50
Graph Break Reason: Encountered gr

  8. [16x] Graph break from `Tensor.item()`, consider setting:
    torch._dynamo.config.capture_scalar_outputs 

  9. [15x] Graph break in user code at transformers/models/fastspeech2_conformer/modeling_fastspeech2_conformer

 10. [12x] Graph break (user stack suppressed due to duplicate graph break) in user code at torch/utils/hooks.p

 11. [12x] Graph break in user code at transformers/utils/import_utils.py:1487
Graph Break Reason: Encountered 

 12. [10x] Graph break in user code at transformers/utils/import_utils.py:1500
Graph Break Reason: Encountered 

 13. [10x] Graph break in user code at transformers/utils/import_utils.py:1502
Graph Break Reason: Encountered 

 14. [6x] Graph break (user stack suppressed due to duplicate graph break) in user code at torch/nn/modules/li

 15. [4x] Graph break in user code at transformers/models/aria/modeling_aria.py:272
Graph Break Reason: Encoun
