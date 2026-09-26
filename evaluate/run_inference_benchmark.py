# -*- coding: utf-8 -*-
"""
Inference benchmark (Section 2.7 clinical-deployment considerations → measured data source for Section 4.3)
================================================================
Purpose: to provide measured values of single-sample inference latency (GPU/CPU) and model size for the paper,
      filling the [to be added] placeholder in the original Section 2.7.

Reference script: 04_five-fold_cross-validation_and_ensemble_evaluation_paper_2.4-3.2_S3-S5/code/
          run_5fold_cv_for_best_model_cnn_model_v3_7_d169_group_comparison.py
          (model construction and checkpoint loading logic are identical to its test_ensemble)

Benchmark protocol:
  - Model: Run 128 best configuration (densenet169 + axial attention, dropout=0)
  - Input: batch=1, 1×182×218×182 (same dimensions as real test-set samples, random tensor,
          timing only the model forward computation, excluding data I/O and preprocessing)
  - GPU: 10 warm-up runs + 30 timed runs (single fold); 5 warm-up runs + 20 timed runs (ensemble)
  - CPU: 2 warm-up runs + 3 timed runs (single fold); 2 timed runs (ensemble)
  - Model size: parameter count × 4 bytes (FP32), plus the actual size of the checkpoint files
Output:
  <output_dir>/benchmark_results.json      (output_dir comes from config.yaml; defaults to results/)
  <output_dir>/benchmark_report.txt
Note: the paper's §4.3 value is the copy archived under reference_results/inference_benchmark/.
      The results/ copy is deliberately not shipped — latency is a hardware snapshot whose value
      varies with GPU thermal state, clock, driver and background load (see README §4.4).
"""
import os
import sys
import json
import time
import statistics
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# ==================== path configuration ====================#
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# The model definition reuses the original training script under train/legacy (that file is an archive; the path needs no change)
LEGACY_DIR = os.path.join(REPO_ROOT, "03_training", "legacy")
sys.path.insert(0, LEGACY_DIR)

from cnn_model_v3_7_d169_group_comparison import (
    NUM_CLASSES, parse_config_from_string
)
from config import load_config, resolve_path

_cfg = load_config()

BEST_RUN_ID = 128
# Both the grid-search results and the 5-fold weights are resolved from config.yaml; no dependency on the author's local paths
RESULTS_CSV = resolve_path(_cfg, "data.grid_results")
WEIGHTS_DIR = resolve_path(_cfg, "weights_dir")
CKPTS = [os.path.join(WEIGHTS_DIR, f"fold_{i}_best_geo.pth") for i in range(1, 6)]

OUT_DIR = _cfg["output_dir"]
os.makedirs(OUT_DIR, exist_ok=True)

INPUT_SHAPE = (1, 1, 182, 218, 182)   # 1 sample, single channel, MNI152 1mm whole brain
SEED = 42


# ==================== utility functions ====================#
def load_config(run_id):
    df = pd.read_csv(RESULTS_CSV)
    row = df[df["run_id"] == run_id]
    if row.empty:
        raise ValueError(f"run_id={run_id} not found")
    return parse_config_from_string(row.iloc[0]["config"])


def build_model(config, target_device):
    """Keep the construction logic identical to test_ensemble in the run_5fold_cv script"""
    from cnn_model_v3_7_d169_group_comparison import get_model
    return get_model(config["model_name"], NUM_CLASSES, target_device,
                     config["dropout_rate"], config["attention_type"])


def load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception:
        if hasattr(torch.serialization, "add_safe_globals"):
            import numpy.core.multiarray  # noqa
            torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
        return torch.load(path, map_location=map_location, weights_only=False)


def load_ensemble(config, target_device):
    """Load the 5-fold ensemble models and return the list of models"""
    models = []
    for path in CKPTS:
        model = build_model(config, target_device)
        ckpt = load_checkpoint(path, map_location=target_device)
        state_dict = ckpt["model_state_dict"]
        if hasattr(model, "base_model") and not any(k.startswith("base_model.") for k in state_dict):
            model.base_model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    return models


def time_forward(model_or_list, x, device, warmup, reps):
    """Time the forward pass for a single fold (pass one model) or the ensemble (pass a list of models); returns a list of times in ms"""
    if not isinstance(model_or_list, (list, tuple)):
        model_or_list = [model_or_list]

    with torch.no_grad():
        for _ in range(warmup):
            for m in model_or_list:
                m(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for m in model_or_list:
                m(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return times


def model_size_mb(model):
    n_params = sum(p.numel() for p in model.parameters())
    n_buffers = sum(b.numel() for b in model.buffers())
    return n_params, n_buffers, (n_params + n_buffers) * 4 / (1024 ** 2)


# ==================== main flow ====================#
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = False  # Disable cudnn auto-tuning; measure the steady state rather than the optimum

    config = load_config(BEST_RUN_ID)
    print(f"[INFO] config: {config}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "run_id": BEST_RUN_ID,
        "config": config,
        "input_shape": list(INPUT_SHAPE),
        "protocol": {"batch_size": 1, "note": "model forward computation only, excluding data I/O and preprocessing"},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu": platform.processor(),
            "torch_num_threads": torch.get_num_threads(),
        },
    }

    # ---------- 1. model size ----------
    cpu_device = torch.device("cpu")
    models_cpu = load_ensemble(config, cpu_device)
    n_params, n_buffers, size_mb = model_size_mb(models_cpu[0])
    ckpt_sizes = [os.path.getsize(p) / (1024 ** 2) for p in CKPTS]
    report["model_size"] = {
        "single_model_params": int(n_params),
        "single_model_buffers": int(n_buffers),
        "single_model_fp32_mb": round(size_mb, 2),
        "single_checkpoint_file_mb_mean": round(float(np.mean(ckpt_sizes)), 2),
        "ensemble_5fold_total_mb": round(size_mb * 5, 2),
    }
    print(f"[INFO] single-fold model: {n_params:,} parameters, FP32 size {size_mb:.1f} MB; "
          f"5-fold ensemble total {size_mb*5:.1f} MB")

    x_cpu = torch.rand(*INPUT_SHAPE)

    # ---------- 2. CPU inference ----------
    print("[INFO] CPU inference benchmark (slower, please wait)...")
    single_cpu = time_forward(models_cpu[0], x_cpu, cpu_device, warmup=2, reps=3)
    ens_cpu = time_forward(models_cpu, x_cpu, cpu_device, warmup=0, reps=2)
    report["cpu"] = {
        "single_fold_ms": [round(t, 1) for t in single_cpu],
        "single_fold_mean_s": round(statistics.mean(single_cpu) / 1000, 3),
        "ensemble_5fold_mean_s": round(statistics.mean(ens_cpu) / 1000, 3),
    }
    print(f"  single fold: {report['cpu']['single_fold_mean_s']} s | "
          f"ensemble: {report['cpu']['ensemble_5fold_mean_s']} s")
    del models_cpu
    import gc; gc.collect()

    # ---------- 3. GPU inference ----------
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        models_gpu = load_ensemble(config, gpu_device)
        x_gpu = x_cpu.to(gpu_device)
        print("[INFO] GPU inference benchmark...")
        single_gpu = time_forward(models_gpu[0], x_gpu, gpu_device, warmup=10, reps=30)
        ens_gpu = time_forward(models_gpu, x_gpu, gpu_device, warmup=5, reps=20)
        report["gpu"] = {
            "single_fold_ms_all": [round(t, 1) for t in single_gpu],
            "single_fold_mean_ms": round(statistics.mean(single_gpu), 1),
            "single_fold_std_ms": round(statistics.stdev(single_gpu), 1),
            "ensemble_5fold_mean_ms": round(statistics.mean(ens_gpu), 1),
            "ensemble_5fold_std_ms": round(statistics.stdev(ens_gpu), 1),
        }
        print(f"  single fold: {report['gpu']['single_fold_mean_ms']}±"
              f"{report['gpu']['single_fold_std_ms']} ms | "
              f"ensemble: {report['gpu']['ensemble_5fold_mean_ms']}±"
              f"{report['gpu']['ensemble_5fold_std_ms']} ms")
        del models_gpu
        gc.collect()
        torch.cuda.empty_cache()

    # ---------- 4. save ----------
    json_path = os.path.join(OUT_DIR, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    txt_path = os.path.join(OUT_DIR, "benchmark_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Inference benchmark report (data source for Section 4.3 of the paper)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Time: {report['timestamp']}\n")
        f.write(f"Model: {config['model_name']} + {config['attention_type']} attention, "
                f"Run {BEST_RUN_ID}\n")
        f.write(f"Input: {INPUT_SHAPE} (single sample, FP32, random tensor, excluding I/O and preprocessing)\n\n")
        f.write(f"Single-fold model parameter count: {n_params:,}\n")
        f.write(f"Single-fold model size (FP32): {size_mb:.1f} MB\n")
        f.write(f"5-fold ensemble total size: {size_mb*5:.1f} MB\n\n")
        f.write(f"CPU single-sample inference (single fold): {report['cpu']['single_fold_mean_s']} s\n")
        f.write(f"CPU single-sample inference (ensemble): {report['cpu']['ensemble_5fold_mean_s']} s\n")
        if "gpu" in report:
            f.write(f"GPU single-sample inference (single fold): {report['gpu']['single_fold_mean_ms']}±"
                    f"{report['gpu']['single_fold_std_ms']} ms\n")
            f.write(f"GPU single-sample inference (ensemble): {report['gpu']['ensemble_5fold_mean_ms']}±"
                    f"{report['gpu']['ensemble_5fold_std_ms']} ms\n")
        f.write(f"\nGPU: {report['environment']['gpu']}\n")
        f.write(f"torch_num_threads(CPU): {report['environment']['torch_num_threads']}\n")
    print(f"[INFO] results saved: {json_path}\n[INFO] report saved: {txt_path}")


if __name__ == "__main__":
    main()
