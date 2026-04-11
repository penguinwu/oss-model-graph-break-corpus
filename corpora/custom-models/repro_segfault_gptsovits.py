"""Minimal repro: dynamo.explain segfaults on GPT-SoVITS VITS model.

Segfault occurs in torch._dynamo.guards.__init__ during build_guards
(profile_guard_manager). Eager mode works fine.

Tested on: torch 2.12.0.dev20260407+cu128, Python 3.12
Works correctly on: torch 2.8, 2.9, 2.10 (4 graph breaks, no crash)

The model is a VITS (Variational Inference Text-to-Speech) architecture
from GPT-SoVITS. The graph breaks come from Tensor.item() calls in
WaveNet's fused_add_tanh_sigmoid_multiply.
"""
import faulthandler
faulthandler.enable()

import sys
import types
import importlib
import torch
import torch._dynamo as dynamo
from pathlib import Path

# --- Download model source files ---
PROXY = "http://localhost:7824/fetch"  # Remove if not behind proxy
GH_RAW = "https://raw.githubusercontent.com"
SRC_DIR = Path("/tmp/gptsovits_repro_src")

def download(url, dest):
    import urllib.request
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    if url is None:
        dest.touch()
        return
    # Direct download (remove proxy wrapper if not needed)
    fetch_url = f"{PROXY}?url={url}" if PROXY else url
    urllib.request.urlretrieve(fetch_url, dest)

FILES = {
    "gptsovits/__init__.py": None,
    "gptsovits/module/__init__.py": None,
    "gptsovits/module/models.py": f"{GH_RAW}/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/module/models.py",
    "gptsovits/module/commons.py": f"{GH_RAW}/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/module/commons.py",
    "gptsovits/module/modules.py": f"{GH_RAW}/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/module/modules.py",
    "gptsovits/module/attentions.py": f"{GH_RAW}/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/module/attentions.py",
    "gptsovits/module/transforms.py": f"{GH_RAW}/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/module/transforms.py",
}

for local, url in FILES.items():
    download(url, SRC_DIR / local)

# --- Mock heavy dependencies ---
for name in ["f5_tts", "module.mrte_model", "module.quantize", "text", "torchmetrics"]:
    sys.modules[name] = types.ModuleType(name)

# f5_tts.model.DiT mock
f5_model = types.ModuleType("f5_tts.model")
f5_dit = types.ModuleType("f5_tts.model.DiT")
class FakeDiT(torch.nn.Module):
    def __init__(self, *a, **kw): super().__init__()
    def forward(self, x): return x
f5_dit.DiT = FakeDiT
sys.modules["f5_tts.model"] = f5_model
sys.modules["f5_tts.model.DiT"] = f5_dit

# module.quantize mock
quantize = types.ModuleType("module.quantize")
class FakeResidualVQ(torch.nn.Module):
    def __init__(self, *a, **kw):
        super().__init__()
        self.n_q = kw.get("num_quantizers", 1)
    def forward(self, x):
        return x, torch.zeros(1), torch.zeros(self.n_q, x.shape[0], x.shape[-1]).long()
quantize.ResidualVectorQuantizer = FakeResidualVQ
sys.modules["module.quantize"] = quantize

# --- Import and create model ---
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "gptsovits"))

from gptsovits.module.models import SynthesizerTrn

model = SynthesizerTrn(
    spec_channels=513, segment_size=20480,
    inter_channels=192, hidden_channels=192, filter_channels=768,
    n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1,
    resblock="1",
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates=[8, 8, 2, 2],
    upsample_initial_channel=512,
    upsample_kernel_sizes=[16, 16, 4, 4],
    n_speakers=0, gin_channels=512, use_sdp=True,
    semantic_frame_rate="50hz", version="v2",
)
model.eval()
print(f"Model created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# --- Generate inputs ---
T_ssl, T_text = 32, 16
inputs = {
    "ssl": torch.randn(2, 768, T_ssl),
    "y": torch.randn(2, 704, T_ssl),
    "y_lengths": torch.tensor([T_ssl, T_ssl]),
    "text": torch.randint(0, 28, (2, T_text)),
    "text_lengths": torch.tensor([T_text, T_text]),
}

# --- Eager works fine ---
print("Testing eager...")
with torch.no_grad():
    model.infer(**inputs)
print("Eager OK")

# --- dynamo.explain segfaults ---
print("Testing dynamo.explain (this segfaults on nightly)...")
dynamo.reset()
explanation = dynamo.explain(model.infer)(**inputs)
print(f"Graph breaks: {explanation.graph_break_count}")
