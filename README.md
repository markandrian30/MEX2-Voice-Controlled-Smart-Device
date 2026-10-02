# Voice Command Module — CRNN

Current checkpoint: **bestsynth** (provisional).

## Run

```bash
git clone https://github.com/markandrian30/vcm-crnn.git
cd vcm-crnn
sudo apt install -y ffmpeg espeak-ng libportaudio2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Place your `music/` and `responses/` folders beside `main.py`. Say **Hello Kibo** to start; **Sagittarius** to exit. Enable I²C for the OLED.

## 1. Model

![Model and example hardware](docs/model.png)

**CNN + bidirectional GRU:** 128 hidden units per direction, dropout 0.35, 987,873 parameters. The current checkpoint has **33 outputs**: 31 commands + wake/exit (correcting the image’s output count).

[Best model](model/best_model.pt)

## 2. Dataset

Main dataset: [airimonda/ai231-me2-voice-commands on Hugging Face](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands). **Preprocessing was performed** to match the command phrases and slot values, exclude overlapping source filenames, and reserve holdout speakers; GitHub synthetic recordings were also included.

The following **prepared CRNN4 subset** is for the later model update; current bestsynth uses the earlier training split.

| Split | GitHub | HF | Total |
|---|---:|---:|---:|
| Training | 14,070 | 316 | 14,386 |
| Validation | 1,806 | 176 | 1,982 |
| Test | 1,614 | 56 | 1,670 |
| **Total** | **17,490** | **548** | **18,038** |

**8.90 hours · 252 speaker IDs · 19 intents · 31 command classes.** Counts exclude 1,044 wake/exit recordings. [Processed manifest](data/command_manifest.csv).

## 3. Training on A100

| Item | bestsynth |
|---|---|
| GPU | 1 × A100-SXM4, 40 GB (DGX2, GPU 6) |
| Loss / optimizer | Cross-entropy, smoothing 0.05 / AdamW |
| Train / validation / test speakers | 122 / 15 / 15 |
| Tuning | 16 trials; selected trial 13 by validation macro F1; seed 42 |
| GRU / dropout | 128 per direction / 0.35 |
| Learning rate / weight decay | Initial 0.001; checkpoint 0.0005 / 0.0001 |
| LR decay / early stopping | Halve after 4 / stop after 10 epochs without validation F1 improvement |
| Checkpoint | Epoch 34/60; validation loss 0.1232 |
| GPU time | 2.68 A100 GPU-hours (author-reported) |

[Training logs and configuration](reports/training).

## 4. Validation on Raspberry Pi 5

Direct-file benchmark; reject confidence **below 25%**.

**Full holdout — 202 recordings**

| Item | Value |
|---|---|
| Command / intent accuracy | **83.66% / 84.65%** |
| False-accept rate | **25% (4/16)** |
| Inference p95 / mean RTF | **57.47 ms / 0.01166** |
| Runtime | PyTorch · 4 threads · Raspberry Pi 5 |

**Phrase-matched holdout — 183 recordings (167 commands + 16 out-of-scope)**

| Item | Value |
|---|---|
| Command / intent accuracy | **90.16% / 91.26%** |
| False-accept rate | **25% (4/16)** |
| Inference p95 / mean RTF | **57.47 ms / 0.01169** |
| Runtime | PyTorch · 4 threads · Raspberry Pi 5 |

```bash
python benchmark.py --min-confidence 0.25
```

Accuracy includes correct out-of-scope rejections. Timing covers features + inference, not microphone/wake/device latency. **Exploratory results:** threshold chosen after inspecting holdout; `s10` overlaps bestsynth training speakers. [Detailed reports](reports/pi).
