"""
WaveDoc — Colab launcher.

Paste into a Colab cell:

    !python run_colab.py

or import and call start().

Copies weights and datasets out of Drive, boots the API, and opens a
public tunnel. Cloudflare needs no account; ngrok is used if a token
is supplied.
"""
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DRIVE = Path("/content/drive/MyDrive/BEACON")
WEIGHTS = ROOT / "weights"
DATA = ROOT / "data"

# candidate source → destination filename
WEIGHT_SOURCES = {
    "MIT-BIH_ECG_Arrhythmia.pt": [
        DRIVE / "FINAL_MODELS/MIT-BIH_ECG_Arrhythmia.pt",
        DRIVE / "checkpoints/mitbih/E_WaveDoc_Full_BEST_ep041.pt",
    ],
    "PTB-XL_ECG_12lead.pt": [
        DRIVE / "FINAL_MODELS/PTB-XL_ECG_12lead.pt",
        DRIVE / "results/ptbxl_v2_BEST.pt",
    ],
    "SleepEDF_EEG_SleepStage.pt": [
        DRIVE / "FINAL_MODELS/SleepEDF_EEG_SleepStage.pt",
        DRIVE / "results/SleepEDF_BEST.pt",
    ],
    "CHBMIT_EEG_Seizure.pt": [
        DRIVE / "FINAL_MODELS/CHBMIT_EEG_Seizure.pt",
        DRIVE / "results/CHBMIT_BEST.pt",
    ],
    "CAMUS_Ultrasound_Segmentation.pt": [
        DRIVE / "FINAL_MODELS/CAMUS_Ultrasound_Segmentation.pt",
        DRIVE / "results/USG_SEG_BEST.pt",
    ],
}

DATA_SOURCES = {
    "ecg_mitbih.pt": [DRIVE / "data/processed/ecg_mitbih.pt"],
    "ecg_ptbxl.pt": [DRIVE / "data/processed/ecg_ptbxl.pt"],
    "eeg_sleepedf.pt": [DRIVE / "data/processed/eeg_sleepedf.pt"],
    "eeg_chbmit.pt": [DRIVE / "data/processed/eeg_chbmit.pt"],
    "ultrasound_camus_seg_test.pt": [
        DRIVE / "data/processed/ultrasound_camus_seg_test.pt"],
}


def stage(sources, dest_dir, label):
    dest_dir.mkdir(parents=True, exist_ok=True)
    found = 0
    for name, candidates in sources.items():
        target = dest_dir / name
        if target.exists():
            found += 1
            continue
        for src in candidates:
            if src.exists():
                shutil.copy(src, target)
                mb = target.stat().st_size / 1024 ** 2
                print(f"    {name}  ({mb:.0f} MB)")
                found += 1
                break
        else:
            print(f"    {name}  — not found")
    print(f"  {label}: {found}/{len(sources)}")
    return found


def install():
    print("  installing dependencies…")
    subprocess.run(
        ["pip", "install", "-q", "fastapi", "uvicorn", "python-multipart"],
        check=False)


def start_server(port=8000):
    env = dict(os.environ)
    env["WAVEDOC_MODELS"] = str(WEIGHTS)
    env["WAVEDOC_DATA"] = str(DATA)
    log = open("/content/wavedoc_server.log", "w")
    proc = subprocess.Popen(
        ["python", str(ROOT / "backend" / "app.py"),
         "--models-dir", str(WEIGHTS),
         "--data-dir", str(DATA),
         "--port", str(port)],
        stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT / "backend"))
    return proc


def wait_for_server(port=8000, timeout=90):
    for _ in range(timeout):
        r = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             f"http://localhost:{port}/api/health"],
            capture_output=True, text=True)
        if r.stdout.strip() == "200":
            return True
        time.sleep(1)
    return False


def tunnel_cloudflare(port=8000):
    binary = Path("/content/cloudflared")
    if not binary.exists():
        subprocess.run(
            ["wget", "-q",
             "https://github.com/cloudflare/cloudflared/releases/latest/"
             "download/cloudflared-linux-amd64",
             "-O", str(binary)], check=True)
        binary.chmod(0o755)
    log_path = "/content/wavedoc_tunnel.log"
    subprocess.Popen([str(binary), "tunnel", "--url",
                      f"http://localhost:{port}"],
                     stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    for _ in range(30):
        time.sleep(1)
        text = Path(log_path).read_text(errors="ignore")
        found = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", text)
        if found:
            return found[0]
    return None


def tunnel_ngrok(port=8000, token=None):
    subprocess.run(["pip", "install", "-q", "pyngrok"], check=False)
    from pyngrok import conf, ngrok
    if token:
        conf.get_default().auth_token = token
    ngrok.kill()
    return ngrok.connect(port, "http").public_url


def start(port=8000, ngrok_token=None):
    print("\nWaveDoc")
    print("=" * 54)

    try:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
    except ImportError:
        print("  not running in Colab — using local paths")

    install()

    print("\n  staging weights…")
    n_w = stage(WEIGHT_SOURCES, WEIGHTS, "weights")
    print("\n  staging datasets…")
    stage(DATA_SOURCES, DATA, "datasets")

    if n_w == 0:
        print("\n  No weights found. Check that FINAL_MODELS exists in Drive,")
        print("  or edit WEIGHT_SOURCES in run_colab.py to match your paths.")
        return None

    print("\n  starting API…")
    subprocess.run(["pkill", "-f", "app.py"], check=False)
    subprocess.run(["pkill", "-f", "cloudflared"], check=False)
    time.sleep(2)
    start_server(port)

    if not wait_for_server(port):
        print("  server did not come up — see /content/wavedoc_server.log")
        subprocess.run(["tail", "-25", "/content/wavedoc_server.log"])
        return None
    print("  API is up")

    print("  opening tunnel…")
    url = (tunnel_ngrok(port, ngrok_token) if ngrok_token
           else tunnel_cloudflare(port))

    print("\n" + "=" * 54)
    if url:
        print(f"  WaveDoc is live at\n  {url}")
    else:
        print("  Tunnel did not start — see /content/wavedoc_tunnel.log")
    print("=" * 54 + "\n")
    return url


if __name__ == "__main__":
    start()
