# WaveDoc — phase-coherent diagnosis across bioelectric signals

**Project BEACON** (Bioelectric Environment-Assisted COherence Network) ·
Group 26 · B.E. Artificial Intelligence & Machine Learning ·
Acharya Institute of Technology · Guide: Ms. Surbhi

WaveDoc keeps the *phase* of a signal, not only its amplitude. Signals are turned
into analytic form with the Hilbert transform and passed through complex-valued
**ENAQT layers** (Environment-Assisted Quantum Transport) with a learnable
decoherence parameter γ, trained with a tunnelling-augmented optimiser
(**TunnelOptim**). One mechanism is applied to ECG, EEG and cardiac ultrasound.

## Results (held-out test sets)

| Dataset | Task | Metric | WaveDoc | Published | Δ |
|---|---|---|---|---|---|
| MIT-BIH | 5-class beat | Macro F1 | 0.9441 | 0.849 (Hannun 2019) | +0.095 |
| PTB-XL | 5 superclasses | Macro AUC | 0.8344 | 0.929 (Strodthoff 2020) | −0.095 |
| Sleep-EDF | 5 sleep stages | Macro F1 | 0.9336 | 0.790 (Phan 2022) | +0.144 |
| CHB-MIT | seizure / normal | AUC | 0.9980 | 0.871 (Gabeff 2021) | +0.127 |
| CAMUS | LV segmentation | Dice | 0.9434 | 0.939 (Leclerc 2019) | +0.004 |

PTB-XL is trained single-label (dominant superclass) while the published baseline is
multi-label; CAMUS three-structure mean Dice is 0.9110 (Residual Attention U-Net 0.9127).
Ablation on MIT-BIH: complex-valued weights provide the gain (0.9267 → 0.9444);
TunnelOptim alone makes no significant difference. See `docs/figures/`.

## Repository layout

```
backend/            FastAPI server + model definitions (models.py)
frontend/           dashboard, model pages, benchmark page (plain HTML/CSS/JS)
tools/              scripts to pick the best checkpoints and pack them
huggingface_space/  Dockerfile etc. for hosting the app on a Hugging Face Space
run_colab.py        one-command launcher for Google Colab
docs/figures/       result diagrams
weights/  data/     (not in git) put trained weights / sample banks here
```

## Run locally

```bash
pip install -r backend/requirements.txt
# put trained weights in ./weights and sample data in ./data
python backend/app.py            # then open http://localhost:8000
```

Expected file names: `weights/MIT-BIH_ECG_Arrhythmia.pt`, `PTB-XL_ECG_12lead.pt`,
`SleepEDF_EEG_SleepStage.pt`, `CHBMIT_EEG_Seizure.pt`, `CAMUS_Ultrasound_Segmentation.pt`.
`tools/find_best_models.py` (Colab) selects and packs the best checkpoint for each task.

## Datasets (not included — download from their owners)
MIT-BIH Arrhythmia · PTB-XL · Sleep-EDF Expanded · CHB-MIT Scalp EEG (all PhysioNet) · CAMUS.
Please follow each dataset's licence.

## Disclaimer
Research prototype. Not a medical device and not for clinical use.
