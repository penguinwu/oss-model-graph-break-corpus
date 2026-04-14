#!/usr/bin/env python3
"""Investigate and fix dimension-mismatch errors (Cluster D).

For each model:
1. Create model with current worker.py (should succeed)
2. Run forward() — capture exact error + traceback
3. Parse error to identify which dimensions mismatch
4. Trace back to config values that caused it
5. Try systematic config adjustments
6. Report fix as a code snippet for worker.py

Usage:
    python tools/fix_worker.py DeepseekV2Model
    python tools/fix_worker.py --models DeepseekV2Model,Zamba2Model
    python tools/fix_worker.py --all-cluster-d --checkpoint sweep_results/v2.10_full/identify_checkpoint.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import textwrap


INVESTIGATE_TEMPLATE = r'''
import sys, json, torch, traceback, inspect, re, gc
sys.path.insert(0, "sweep")
from worker import create_hf_model, _fix_config, _reduce_model_size
import transformers

model_name = "MODEL_NAME_PLACEHOLDER"
device = "cuda"
B = 2

# Step 1: Get spec from enumerate_all
from models import enumerate_all
all_models = {m["name"]: m for m in enumerate_all()}
spec = all_models.get(model_name)
if not spec:
    print(json.dumps({"status": "skip", "reason": "not in enumerate_all"}))
    sys.exit(0)

# Step 2: Create model
try:
    model, inputs, tok = create_hf_model(spec, device, batch_size=B)
except Exception as e:
    print(json.dumps({"status": "create_error", "error": str(e)[:300]}))
    sys.exit(0)

# Step 3: Collect config info post-creation
config = model.config
config_info = {}
for attr in ["hidden_size", "num_attention_heads", "num_key_value_heads",
              "intermediate_size", "num_hidden_layers", "vocab_size",
              "num_codebooks"]:
    val = getattr(config, attr, None)
    if val is not None:
        config_info[attr] = val if isinstance(val, (int, float, str, bool)) else str(val)
for sub_name in ["text_config", "vision_config", "audio_config", "decoder"]:
    sub = getattr(config, sub_name, None)
    if sub:
        for attr in ["hidden_size", "num_attention_heads", "num_hidden_layers",
                     "num_key_value_heads", "intermediate_size", "patch_size",
                     "image_size", "in_channels"]:
            val = getattr(sub, attr, None)
            if val is not None:
                config_info[f"{sub_name}.{attr}"] = val if isinstance(val, (int, float, str, bool)) else str(val)

# Step 4: Try forward and capture full traceback
model.eval()
if inputs is None:
    print(json.dumps({"status": "no_inputs", "config": config_info}))
    sys.exit(0)

try:
    with torch.no_grad():
        if isinstance(inputs, dict):
            out = model(**inputs)
        else:
            out = model(*inputs)
    print(json.dumps({"status": "ok", "config": config_info}))
    sys.exit(0)
except Exception as e:
    tb = traceback.format_exc()
    frames = tb.strip().split("\n")
    error_frames = [f.strip() for f in frames if "File " in f and "site-packages" in f]

    error_info = {
        "error_type": type(e).__name__,
        "error_msg": str(e)[:300],
        "error_frames": error_frames[-3:],
        "full_traceback_tail": "\n".join(frames[-10:]),
    }

del model
gc.collect()
torch.cuda.empty_cache()

# Step 5: Try systematic fixes
fixes_tried = []
msg = error_info["error_msg"]

def try_fix(fix_desc, config_mods):
    """Try a config modification and see if it fixes the error."""
    fixes_tried.append(fix_desc)
    try:
        config2 = type(config).from_dict(config.to_dict())
        for k, v in config_mods.items():
            if "." in k:
                sub_name, attr = k.split(".", 1)
                sub = getattr(config2, sub_name, None)
                if sub:
                    setattr(sub, attr, v)
            else:
                setattr(config2, k, v)
        # Re-align heads if hidden_size changed
        hs = getattr(config2, "hidden_size", None)
        nh = getattr(config2, "num_attention_heads", None)
        if hs and nh and isinstance(hs, int) and isinstance(nh, int) and hs % nh != 0:
            for h in [16, 8, 4, 2, 1]:
                if hs % h == 0:
                    config2.num_attention_heads = h
                    break
        nkv = getattr(config2, "num_key_value_heads", None)
        nh2 = getattr(config2, "num_attention_heads", None)
        if nkv and nh2 and isinstance(nkv, int) and isinstance(nh2, int) and nh2 % nkv != 0:
            config2.num_key_value_heads = nh2

        model2 = type(config2).from_dict(config2.to_dict())  # validate config
        # Actually create the model
        model_cls = getattr(transformers, spec.get("hf_class") or spec["name"])
        model2 = model_cls(config2).to(device)
        _, inputs2, _ = create_hf_model(spec, device, batch_size=B)
        model2.eval()
        with torch.no_grad():
            if isinstance(inputs2, dict):
                out = model2(**inputs2)
            else:
                out = model2(*inputs2)
        del model2
        gc.collect()
        torch.cuda.empty_cache()
        return True, fix_desc
    except Exception as e2:
        try:
            del model2
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        return False, str(e2)[:150]

# Strategy A: For stride errors, try different hidden_size values
if "stride" in msg.lower():
    for hs in [512, 1024, 2048, 4096]:
        ok, detail = try_fix(f"hidden_size={hs}", {"hidden_size": hs})
        if ok:
            error_info["fix_found"] = f"hidden_size={hs}"
            error_info["fix_details"] = f"Set hidden_size to {hs} for 16-byte stride alignment"
            break

# Strategy B: For list index errors, try more layers
if "list index" in msg:
    for n in [4, 6, 8]:
        mods = {"num_hidden_layers": n}
        # Re-expand layer_types/layers_block_type from original config
        orig_cfg = type(config)()
        for lt_attr in ("layer_types", "layers_block_type"):
            orig_lt = getattr(orig_cfg, lt_attr, None)
            if orig_lt and isinstance(orig_lt, (list, tuple)):
                mods[lt_attr] = list(orig_lt[:n])
        ok, detail = try_fix(f"num_hidden_layers={n}", mods)
        if ok:
            error_info["fix_found"] = f"num_hidden_layers={n}"
            error_info["fix_details"] = f"Increase num_hidden_layers to {n} (2 was too aggressive)"
            break

# Strategy C: For tensor size mismatch, try hidden_size adjustments
if "size of tensor" in msg or "shapes cannot" in msg:
    nums = re.findall(r'\b(\d+)\b', msg)
    error_info["parsed_dims"] = nums
    orig_hs = config_info.get("hidden_size", 768)
    if isinstance(orig_hs, int):
        for hs in [orig_hs * 2, orig_hs // 2, 768, 1024, 512, 256]:
            if hs < 64 or hs > 8192:
                continue
            ok, detail = try_fix(f"hidden_size={hs}", {"hidden_size": hs})
            if ok:
                error_info["fix_found"] = f"hidden_size={hs}"
                error_info["fix_details"] = f"Set hidden_size to {hs} (was {orig_hs})"
                break

# Strategy D: For CUDA errors, try reducing vocab_size and other sizes
if "CUDA error" in msg:
    for mods in [
        {"vocab_size": 100},
        {"num_codebooks": 2},
        {"vocab_size": 100, "num_codebooks": 2},
        {"hidden_size": 256, "intermediate_size": 512},
    ]:
        # Only try mods for attributes that exist
        valid_mods = {k: v for k, v in mods.items() if hasattr(config, k)}
        if not valid_mods:
            continue
        ok, detail = try_fix(str(valid_mods), valid_mods)
        if ok:
            error_info["fix_found"] = str(valid_mods)
            error_info["fix_details"] = f"Config reduction: {valid_mods}"
            break

# Strategy E: For permute errors, try different image/patch sizes
if "permute" in msg or "dimension" in msg.lower():
    vision_cfg = getattr(config, "vision_config", None)
    if vision_cfg:
        for img_sz in [224, 256, 384]:
            ok, detail = try_fix(f"vision.image_size={img_sz}",
                                {"vision_config.image_size": img_sz})
            if ok:
                error_info["fix_found"] = f"vision_config.image_size={img_sz}"
                error_info["fix_details"] = f"Set vision image_size to {img_sz}"
                break

error_info["fixes_tried"] = fixes_tried
error_info["config"] = config_info
error_info["status"] = "fixed" if "fix_found" in error_info else "unfixed"
print(json.dumps(error_info))
'''


def investigate_model(model_name, timeout=120):
    """Investigate a single model's error in an isolated subprocess.

    Returns dict with diagnosis and fix suggestion.
    """
    # Write script to temp file to avoid f-string escaping issues
    script_path = f"/tmp/fix_investigate_{model_name}.py"
    script_content = INVESTIGATE_TEMPLATE.replace("MODEL_NAME_PLACEHOLDER", model_name)
    with open(script_path, "w") as f:
        f.write(script_content)

    try:
        r = subprocess.run(
            ["/home/pengwu/envs/torch210/bin/python", script_path],
            capture_output=True, text=True, timeout=timeout,
            cwd="/home/pengwu/projects/oss-model-graph-break-corpus",
        )
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        return {"status": "no_output", "stdout": r.stdout[-500:], "stderr": r.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", help="Single model name")
    parser.add_argument("--models", help="Comma-separated model names")
    parser.add_argument("--checkpoint", help="Checkpoint JSONL for --all-cluster-d")
    parser.add_argument("--all-cluster-d", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    if args.model:
        names = [args.model]
    elif args.models:
        names = [n.strip() for n in args.models.split(",")]
    elif args.all_cluster_d and args.checkpoint:
        names = []
        seen = set()
        with open(args.checkpoint) as f:
            for line in f:
                d = json.loads(line)
                if d.get("source", "") not in ("hf", "diffusers", "custom"):
                    continue
                if d["status"] not in ("create_error", "eager_error"):
                    continue
                err = d.get("error", "")
                if any(x in err for x in ["tensor a", "strides", "mat1 and mat2",
                                          "shapes cannot", "permute", "CUDA error",
                                          "list index"]):
                    if d["name"] not in seen:
                        seen.add(d["name"])
                        names.append(d["name"])
    else:
        parser.print_help()
        return

    print(f"Investigating {len(names)} models...")
    results = []
    for i, name in enumerate(names):
        print(f"\n[{i+1}/{len(names)}] {name}")
        result = investigate_model(name, timeout=args.timeout)
        result["name"] = name
        results.append(result)

        status = result.get("status", "?")
        if status == "ok":
            print(f"  Already works! (no error)")
        elif status == "fixed":
            print(f"  ✓ FIXED: {result.get('fix_details', '?')}")
        elif status == "unfixed":
            print(f"  ✗ Not fixed. Error: {result.get('error_msg', '?')[:100]}")
            print(f"    Tried: {result.get('fixes_tried', [])}")
            if result.get("parsed_dims"):
                print(f"    Mismatched dims: {result['parsed_dims']}")
            if result.get("error_frames"):
                print(f"    Error location: {result['error_frames'][-1][:120]}")
        elif status == "timeout":
            print(f"  ⏰ Timeout")
        else:
            print(f"  Status: {status}")
            if result.get("error"):
                print(f"  Error: {result['error'][:150]}")

    # Summary
    fixed = [r for r in results if r.get("status") == "fixed"]
    unfixed = [r for r in results if r.get("status") == "unfixed"]
    already_ok = [r for r in results if r.get("status") == "ok"]
    print(f"\n{'='*60}")
    print(f"Already OK: {len(already_ok)}")
    print(f"Fixed: {len(fixed)}")
    print(f"Unfixed: {len(unfixed)}")

    if fixed:
        print(f"\nFixes to apply:")
        for r in fixed:
            print(f"  {r['name']}: {r.get('fix_details')}")

    # Save results
    with open("/tmp/fix_worker_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /tmp/fix_worker_results.json")


if __name__ == "__main__":
    main()
