"""
WaveDoc — API server.

Serves the dashboard and exposes inference endpoints for all five models.

Run:
    python backend/app.py
    python backend/app.py --models-dir /path/to/FINAL_MODELS
"""
import argparse
import io
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from scipy.signal import hilbert

from models import (BEACONForEEG, BEACONForPTBXL, BEACONSegUSG,
                    QuantumBioNet)

# ─────────────────────────────────────────────────────────────
# Paths — override with CLI flags or env vars
# ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("WAVEDOC_MODELS", ROOT / "weights"))
DATA_DIR = Path(os.environ.get("WAVEDOC_DATA", ROOT / "data"))
FRONTEND = ROOT / "frontend"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# Registry — every model the dashboard knows about
# ─────────────────────────────────────────────────────────────
REGISTRY = {
    "mitbih": {
        "title": "Heart rhythm",
        "subtitle": "Single-lead ECG · beat classification",
        "dataset": "MIT-BIH Arrhythmia Database",
        "kind": "signal",
        "weights": ["MIT-BIH_ECG_Arrhythmia.pt", "model_mitbih.pt"],
        "data": ["ecg_mitbih.pt"],
        "shape": [360],
        "shape_text": "360 × 1",
        "channels": 1,
        "channel_text": "1 lead",
        "rate": "360 Hz",
        "duration": "1 second",
        "classes": ["Normal", "PVC — ventricular", "APC — atrial",
                    "RBBB", "LBBB"],
        "builder": lambda: QuantumBioNet(360, 256, 3, 5),
        "arch": {
            "encoder": "none — raw 360-sample window",
            "layers": 3,
            "hidden": 256,
            "params": "~0.5 M",
            "optimizer": "TunnelOptim (lr 1e-3)",
            "epochs": 50,
        },
        "metrics": {"Macro F1": 0.9441, "Accuracy": 0.9808, "AUC": 0.9980},
        "primary": ["Macro F1", 0.9441],
        "baseline": {"name": "Hannun et al. 2019", "metric": "Macro F1",
                     "value": 0.849},
        "gamma": [0.2706, 0.2250, 0.2207],
        "notes": "Accuracy saturates on this benchmark because 68% of beats "
                 "are normal. Macro F1 weights all five classes equally and "
                 "is the metric reported here.",
    },
    "ptbxl": {
        "title": "Cardiac condition",
        "subtitle": "12-lead ECG · superclass diagnosis",
        "dataset": "PTB-XL",
        "kind": "signal",
        "weights": ["PTB-XL_ECG_12lead.pt", "ptbxl_v2_BEST.pt",
                    "model_ptbxl.pt"],
        "data": ["ecg_ptbxl.pt"],
        "shape": [12, 1000],
        "shape_text": "12 × 1000",
        "channels": 12,
        "channel_text": "12 leads",
        "rate": "100 Hz",
        "duration": "10 seconds",
        "flatten": True,
        "classes": ["Normal", "Myocardial infarction", "ST/T change",
                    "Conduction disturbance", "Hypertrophy"],
        "builder": lambda: BEACONForPTBXL(5, 6),
        "arch": {
            "encoder": "Conv1d 12→64→128→256→512, adaptive pool",
            "layers": 6,
            "hidden": 512,
            "params": "~4.5 M",
            "optimizer": "TunnelOptim + mixup + label smoothing 0.1",
            "epochs": 120,
        },
        "metrics": {"Macro F1": 0.6090, "Accuracy": 0.7234, "AUC": 0.8344},
        "primary": ["AUC", 0.8344],
        "baseline": {"name": "Strodthoff et al. 2020", "metric": "Macro AUC",
                     "value": 0.929},
        "gamma": [0.2809, 0.2747, 0.2561, 0.2495, 0.2473, 0.1848],
        "notes": "PTB-XL is formally multi-label. We train a single-label "
                 "formulation using each record's dominant superclass, which "
                 "discards co-occurring diagnoses. The shortfall against the "
                 "ResNet baseline is reported as measured.",
    },
    "sleep": {
        "title": "Sleep stage",
        "subtitle": "EEG · 30-second epoch scoring",
        "dataset": "Sleep-EDF Expanded",
        "kind": "signal",
        "weights": ["SleepEDF_EEG_SleepStage.pt", "SleepEDF_BEST.pt",
                    "model_sleep.pt"],
        "data": ["eeg_sleepedf.pt"],
        "shape": [2, 3000],
        "shape_text": "2 × 3000",
        "channels": 2,
        "channel_text": "2 channels",
        "rate": "100 Hz",
        "duration": "30 seconds",
        "classes": ["Wake", "N1 light", "N2", "N3 deep", "REM"],
        "builder": lambda: BEACONForEEG(2, 5, 6),
        "arch": {
            "encoder": "Two-branch Conv1d — 7px transient + 25px rhythm",
            "layers": 6,
            "hidden": 512,
            "params": "~9 M",
            "optimizer": "TunnelOptim + OneCycle + mixup",
            "epochs": 80,
        },
        "metrics": {"Macro F1": 0.9336, "Accuracy": 0.9345, "AUC": 0.9912},
        "primary": ["Macro F1", 0.9336],
        "baseline": {"name": "Phan et al. 2022", "metric": "Macro F1",
                     "value": 0.790},
        "gamma": [0.2610, 0.2180, 0.1740, 0.1420, 0.1180, 0.0930],
        "notes": "Gains concentrate on N1 and REM — the two stages defined "
                 "by cross-channel synchrony rather than wave amplitude.",
    },
    "seizure": {
        "title": "Seizure detection",
        "subtitle": "EEG · 10-second window screening",
        "dataset": "CHB-MIT Scalp EEG",
        "kind": "signal",
        "weights": ["CHBMIT_EEG_Seizure.pt", "CHBMIT_BEST.pt",
                    "model_chbmit.pt"],
        "data": ["eeg_chbmit.pt"],
        "shape": [18, 2560],
        "shape_text": "18 × 2560",
        "channels": 18,
        "channel_text": "18 channels",
        "rate": "256 Hz",
        "duration": "10 seconds",
        "classes": ["Normal activity", "Seizure"],
        "builder": lambda: BEACONForEEG(18, 2, 6),
        "arch": {
            "encoder": "Two-branch Conv1d — 7px transient + 25px rhythm",
            "layers": 6,
            "hidden": 512,
            "params": "~9 M",
            "optimizer": "TunnelOptim, inverse-frequency^1.5 sampling",
            "epochs": 60,
        },
        "metrics": {"Macro F1": 0.9976, "Accuracy": 0.9992,
                    "AUC": 0.9980, "Seizure recall": 0.9927},
        "primary": ["AUC", 0.9980],
        "baseline": {"name": "Gabeff et al. 2021", "metric": "AUC",
                     "value": 0.871},
        "gamma": [0.3000, 0.2400, 0.1900, 0.1500, 0.1200, 0.0900],
        "notes": "6 of 823 seizure windows missed, 1 false positive across "
                 "7,429 normal windows, against a 9:1 class imbalance.",
    },
    "ultrasound": {
        "title": "Cardiac ultrasound",
        "subtitle": "Echocardiogram · structure segmentation",
        "dataset": "CAMUS",
        "kind": "image",
        "weights": ["CAMUS_Ultrasound_Segmentation.pt", "USG_SEG_BEST.pt",
                    "model_usg.pt"],
        "data": ["ultrasound_camus_seg_test.pt", "usg_seg_test.pt"],
        "shape": [1, 128, 128],
        "shape_text": "128 × 128",
        "channels": 1,
        "channel_text": "1 frame",
        "rate": "—",
        "duration": "single frame",
        "classes": ["Background", "Left ventricle", "Myocardium",
                    "Left atrium"],
        "builder": lambda: BEACONSegUSG(4, 32),
        "arch": {
            "encoder": "4-level U-Net, dual real/imaginary paths",
            "layers": 4,
            "hidden": 256,
            "params": "~7.8 M",
            "optimizer": "AdamW + OneCycle, γ excluded from weight decay",
            "epochs": 120,
        },
        "metrics": {"Dice LV": 0.9434, "Dice Myocardium": 0.8857,
                    "Dice LA": 0.9040, "Dice mean": 0.9110},
        "primary": ["Dice LV", 0.9434],   # same structure as the baseline
        "baseline": {"name": "Leclerc et al. 2019 (LV)", "metric": "Dice",
                     "value": 0.939},
        "gamma": [0.1271, 0.0845, 0.0674, 0.0546],
        "notes": "LV Dice of 0.9434 marginally exceeds the U-Net baseline "
                 "published with the dataset. Official 450/50 patient split.",
    },
}

BENCHMARKS = [
    {"benchmark": "CHB-MIT", "metric": "AUC", "baseline": 0.871,
     "reference": "Gabeff et al. 2021", "wavedoc": 0.9980, "key": "seizure"},
    {"benchmark": "Sleep-EDF", "metric": "Macro F1", "baseline": 0.790,
     "reference": "Phan et al. 2022", "wavedoc": 0.9336, "key": "sleep"},
    {"benchmark": "MIT-BIH", "metric": "Macro F1", "baseline": 0.849,
     "reference": "Hannun et al. 2019", "wavedoc": 0.9441, "key": "mitbih"},
    {"benchmark": "CAMUS — LV", "metric": "Dice", "baseline": 0.939,
     "reference": "Leclerc et al. 2019", "wavedoc": 0.9434,
     "key": "ultrasound"},
    {"benchmark": "CAMUS — mean", "metric": "Dice", "baseline": 0.9127,
     "reference": "Residual Attention U-Net", "wavedoc": 0.9110,
     "key": "ultrasound"},
    {"benchmark": "PTB-XL", "metric": "Macro AUC", "baseline": 0.929,
     "reference": "Strodthoff et al. 2020", "wavedoc": 0.8344,
     "key": "ptbxl"},
]

ABLATION = [
    {"variant": "A — RealNN + Adam", "complex": False, "tunnel": False,
     "gamma": False, "f1": 0.9267, "acc": 0.9728, "auc": 0.9988},
    {"variant": "B — CVNN + Adam", "complex": True, "tunnel": False,
     "gamma": False, "f1": 0.9444, "acc": 0.9813, "auc": 0.9977},
    {"variant": "C — RealNN + TunnelOptim", "complex": False, "tunnel": True,
     "gamma": False, "f1": 0.9291, "acc": 0.9739, "auc": 0.9981},
    {"variant": "D — CVNN + TunnelOptim", "complex": True, "tunnel": True,
     "gamma": False, "f1": 0.9438, "acc": 0.9805, "auc": 0.9977},
    {"variant": "E — QuantumBioNet", "complex": True, "tunnel": True,
     "gamma": True, "f1": 0.9441, "acc": 0.9808, "auc": 0.9980},
]

# ─────────────────────────────────────────────────────────────
# Lazy loading with cache
# ─────────────────────────────────────────────────────────────
_models = {}
_datasets = {}


def _find(candidates, directory):
    for name in candidates:
        p = directory / name
        if p.exists():
            return p
    return None


def get_model(key):
    if key in _models:
        return _models[key]
    cfg = REGISTRY[key]
    path = _find(cfg["weights"], MODELS_DIR)
    if path is None:
        _models[key] = None
        return None
    model = cfg["builder"]()
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False
    _models[key] = model
    print(f"  loaded {key:<12} from {path.name}")
    return model


def get_dataset(key):
    if key in _datasets:
        return _datasets[key]
    path = _find(REGISTRY[key]["data"], DATA_DIR)
    _datasets[key] = torch.load(path, map_location="cpu") if path else None
    if path:
        print(f"  dataset {key:<11} from {path.name}")
    return _datasets[key]


# ─────────────────────────────────────────────────────────────
# Signal helpers
# ─────────────────────────────────────────────────────────────
def normalise(sig):
    mu = sig.mean(axis=-1, keepdims=True)
    sd = sig.std(axis=-1, keepdims=True) + 1e-8
    return (sig - mu) / sd


def to_complex(sig):
    c = hilbert(sig, axis=-1)
    return (torch.tensor(c.real, dtype=torch.float32),
            torch.tensor(c.imag, dtype=torch.float32))


def coerce(arr, want):
    """Pad or crop an array to the shape the model expects."""
    adjusted = False
    if len(want) == 1:
        arr = arr.flatten()
        if arr.shape[0] != want[0]:
            arr = (arr[:want[0]] if arr.shape[0] > want[0]
                   else np.pad(arr, (0, want[0] - arr.shape[0])))
            adjusted = True
    elif len(want) == 2:
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if tuple(arr.shape) != tuple(want):
            fixed = np.zeros(want, dtype=np.float32)
            r, c = min(arr.shape[0], want[0]), min(arr.shape[1], want[1])
            fixed[:r, :c] = arr[:r, :c]
            arr, adjusted = fixed, True
    else:
        while arr.ndim < 3:
            arr = arr[None, ...]
        arr = arr[..., :, :] if arr.ndim == 3 else arr.reshape(1, *arr.shape[-2:])
        if tuple(arr.shape) != tuple(want):
            fixed = np.zeros(want, dtype=np.float32)
            a = [min(arr.shape[i], want[i]) for i in range(3)]
            fixed[:a[0], :a[1], :a[2]] = arr[:a[0], :a[1], :a[2]]
            arr, adjusted = fixed, True
    return arr.astype(np.float32), adjusted


def downsample(arr, target=600):
    """Thin a trace for transport without changing its shape visually."""
    if arr.shape[-1] <= target:
        return arr
    step = arr.shape[-1] // target
    return arr[..., ::step]


def run_signal(model, key, xr, xi):
    cfg = REGISTRY[key]
    if cfg.get("flatten") and xr.dim() == 3:
        xr = xr.view(xr.shape[0], -1)
        xi = xi.view(xi.shape[0], -1)
    with torch.no_grad():
        out = model(xr.to(DEVICE), xi.to(DEVICE))
        prob = torch.softmax(out, 1)[0].cpu().numpy()
    pred = int(prob.argmax())
    return pred, prob


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="WaveDoc API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    ready = {}
    for key in REGISTRY:
        ready[key] = {
            "weights": _find(REGISTRY[key]["weights"], MODELS_DIR) is not None,
            "data": _find(REGISTRY[key]["data"], DATA_DIR) is not None,
        }
    return {"device": str(DEVICE), "models_dir": str(MODELS_DIR),
            "data_dir": str(DATA_DIR), "available": ready}


@app.get("/api/models")
def list_models():
    out = []
    for key, cfg in REGISTRY.items():
        out.append({
            "key": key, "title": cfg["title"], "subtitle": cfg["subtitle"],
            "dataset": cfg["dataset"], "kind": cfg["kind"],
            "shape_text": cfg["shape_text"], "channel_text": cfg["channel_text"],
            "rate": cfg["rate"], "duration": cfg["duration"],
            "classes": cfg["classes"], "primary": cfg["primary"],
            "baseline": cfg["baseline"],
            "has_weights": _find(cfg["weights"], MODELS_DIR) is not None,
            "has_data": _find(cfg["data"], DATA_DIR) is not None,
        })
    return out


@app.get("/api/models/{key}")
def model_detail(key: str):
    if key not in REGISTRY:
        raise HTTPException(404, "unknown model")
    cfg = REGISTRY[key]
    detail = {k: v for k, v in cfg.items()
              if k not in ("builder", "weights", "data")}
    detail["key"] = key
    detail["has_weights"] = _find(cfg["weights"], MODELS_DIR) is not None

    model = get_model(key)
    if model is not None:
        detail["gamma_live"] = [round(v, 4)
                                for v in model.get_gamma_profile().values()]
        g = detail["gamma_live"]
        detail["gamma_monotonic"] = all(g[i] > g[i + 1]
                                        for i in range(len(g) - 1))
    return detail


@app.get("/api/benchmarks")
def benchmarks():
    rows = []
    for b in BENCHMARKS:
        d = dict(b)
        d["delta"] = round(b["wavedoc"] - b["baseline"], 4)
        d["leads"] = d["delta"] > 0
        rows.append(d)
    return {"rows": rows,
            "wins": sum(1 for r in rows if r["leads"]),
            "total": len(rows)}


@app.get("/api/ablation")
def ablation():
    return ABLATION


@app.get("/api/sample/{key}")
def sample(key: str, cls: int = -1):
    """Return a random recording, optionally restricted to one class."""
    if key not in REGISTRY:
        raise HTTPException(404, "unknown model")
    cfg = REGISTRY[key]
    data = get_dataset(key)
    model = get_model(key)
    if data is None:
        raise HTTPException(503, f"dataset for {key} not available")
    if model is None:
        raise HTTPException(503, f"weights for {key} not available")

    XR, XI = data["xr"], data["xi"]
    labels = data.get("y")

    if cls >= 0 and labels is not None:
        idx_pool = (labels == cls).nonzero().flatten()
        if len(idx_pool) == 0:
            raise HTTPException(400, "no samples in that class")
        j = int(idx_pool[random.randrange(len(idx_pool))])
    else:
        j = random.randrange(len(XR))

    truth = int(labels[j]) if labels is not None else None
    xr, xi = XR[j:j + 1], XI[j:j + 1]

    if cfg["kind"] == "signal":
        pred, prob = run_signal(model, key, xr, xi)
        trace = downsample(XR[j].numpy())
        if trace.ndim == 1:
            trace = trace[None, :]
        return {
            "index": j, "kind": "signal",
            "prediction": pred, "predicted_class": cfg["classes"][pred],
            "confidence": float(prob[pred]),
            "probabilities": [float(p) for p in prob],
            "classes": cfg["classes"],
            "truth": truth,
            "truth_class": cfg["classes"][truth] if truth is not None else None,
            "correct": (pred == truth) if truth is not None else None,
            "channels": [[round(float(v), 4) for v in ch]
                         for ch in trace[:4]],
            "n_channels": int(trace.shape[0]),
            "phase_map": phase_grid(model),
        }

    # segmentation
    with torch.no_grad():
        out = model(xr.to(DEVICE), xi.to(DEVICE))
        seg = out.argmax(1)[0].cpu().numpy()
        pmap = model.get_phase_spatial(xr.to(DEVICE),
                                       xi.to(DEVICE)).cpu().numpy()
    frame = XR[j][0].numpy()
    result = {
        "index": j, "kind": "image", "classes": cfg["classes"],
        "frame": [[round(float(v), 3) for v in row] for row in frame[::2, ::2]],
        "mask": [[int(v) for v in row] for row in seg[::2, ::2]],
        "phase": [[round(float(v), 3) for v in row] for row in pmap],
        "coverage": [round(float((seg == c).sum() / seg.size), 4)
                     for c in range(4)],
    }
    if "mask" in data:
        gt = data["mask"][j].numpy()
        dice = []
        for c in range(1, 4):
            p, t = (seg == c), (gt == c)
            inter, union = (p & t).sum(), p.sum() + t.sum()
            dice.append(round(float(2 * inter / (union + 1e-9)), 4))
        result["dice"] = dice
    return result


@app.post("/api/predict/{key}")
async def predict(key: str, file: UploadFile = File(...)):
    """Classify an uploaded csv / txt / npy recording."""
    if key not in REGISTRY:
        raise HTTPException(404, "unknown model")
    cfg = REGISTRY[key]
    model = get_model(key)
    if model is None:
        raise HTTPException(503, f"weights for {key} not available")

    raw = await file.read()
    name = (file.filename or "").lower()
    try:
        if name.endswith(".npy"):
            arr = np.load(io.BytesIO(raw))
        else:
            text = raw.decode("utf-8", errors="replace")
            arr = np.loadtxt(io.StringIO(text), delimiter=",")
    except Exception as exc:
        raise HTTPException(400, f"could not read that file — {exc}")

    arr, adjusted = coerce(np.asarray(arr, dtype=np.float32), cfg["shape"])

    if cfg["kind"] == "signal":
        r, i = to_complex(normalise(arr))
        xr, xi = r.unsqueeze(0), i.unsqueeze(0)
        pred, prob = run_signal(model, key, xr, xi)
        trace = downsample(arr if arr.ndim > 1 else arr[None, :])
        return {
            "kind": "signal", "adjusted": adjusted,
            "prediction": pred, "predicted_class": cfg["classes"][pred],
            "confidence": float(prob[pred]),
            "probabilities": [float(p) for p in prob],
            "classes": cfg["classes"],
            "channels": [[round(float(v), 4) for v in ch] for ch in trace[:4]],
            "n_channels": int(trace.shape[0]),
            "phase_map": phase_grid(model),
        }

    r, i = to_complex(arr)
    xr, xi = r.unsqueeze(0), i.unsqueeze(0)
    with torch.no_grad():
        out = model(xr.to(DEVICE), xi.to(DEVICE))
        seg = out.argmax(1)[0].cpu().numpy()
        pmap = model.get_phase_spatial(xr.to(DEVICE),
                                       xi.to(DEVICE)).cpu().numpy()
    return {
        "kind": "image", "adjusted": adjusted, "classes": cfg["classes"],
        "frame": [[round(float(v), 3) for v in row] for row in arr[0][::2, ::2]],
        "mask": [[int(v) for v in row] for row in seg[::2, ::2]],
        "phase": [[round(float(v), 3) for v in row] for row in pmap],
        "coverage": [round(float((seg == c).sum() / seg.size), 4)
                     for c in range(4)],
    }


def phase_grid(model, size=64):
    """Coarse phase map for transport to the browser."""
    try:
        pm = model.get_phase().detach().cpu().numpy()
    except Exception:
        return None
    if pm.ndim != 2:
        return None
    step_r = max(1, pm.shape[0] // size)
    step_c = max(1, pm.shape[1] // size)
    small = pm[::step_r, ::step_c][:size, :size]
    return [[round(float(v), 3) for v in row] for row in small]


# ─────────────────────────────────────────────────────────────
# Static frontend
# ─────────────────────────────────────────────────────────────
if FRONTEND.exists():
    app.mount("/css", StaticFiles(directory=FRONTEND / "css"), name="css")
    app.mount("/js", StaticFiles(directory=FRONTEND / "js"), name="js")

    @app.get("/")
    @app.get("/index.html")
    def index():
        return FileResponse(FRONTEND / "index.html")

    @app.get("/models.html")
    def models_page():
        return FileResponse(FRONTEND / "models.html")

    @app.get("/compare.html")
    def compare_page():
        return FileResponse(FRONTEND / "compare.html")


def main():
    global MODELS_DIR, DATA_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    MODELS_DIR = Path(args.models_dir)
    DATA_DIR = Path(args.data_dir)

    print("\nWaveDoc API")
    print(f"  device      {DEVICE}")
    print(f"  weights     {MODELS_DIR}")
    print(f"  datasets    {DATA_DIR}")
    print("  preloading models…")
    for key in REGISTRY:
        get_model(key)
        get_dataset(key)
    print(f"\n  serving on http://{args.host}:{args.port}\n")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
