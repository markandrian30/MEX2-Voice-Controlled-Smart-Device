# Voice Controlled Smart Device

## Run

```bash
git clone https://github.com/markandrian30/MEX2-Voice-Controlled-Smart-Device.git
cd MEX2-Voice-Controlled-Smart-Device
sudo apt install -y ffmpeg espeak-ng libportaudio2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

- Place `music/` and `responses/` beside `main.py`.
- Say **Hello Kibo before each command**; **Sagittarius** to exit.
- Accepts confidence **above 30%**; otherwise says "Command not recognized" during an active command session.
- **Hardware:** Raspberry Pi 5 with a suitable power supply, USB microphone, and USB speaker (or a powered speaker through a USB audio adapter).
- **LEDs:** Connect separate red, green, and blue LEDs on a breadboard, each with a 330-ohm series resistor and jumper wires, as shown below.
- **Display:** Connect a 3.3 V-compatible SSD1306 128 x 64 I2C OLED at address `0x3C`. Enable I2C using `sudo raspi-config`.

### Hardware wiring

Power off the Pi before wiring. Pin numbers below are physical header pins; GPIO labels use BCM numbering.

```mermaid
flowchart LR
    MIC[USB microphone] -->|USB| PI[Raspberry Pi 5]
    PI -->|USB audio| SPK[USB speaker / audio adapter]
    PSU[Pi power supply] -->|USB-C power| PI

    subgraph HEADER[Pi 40-pin header]
        VCC[3.3 V - pin 1]
        SDA[GPIO2 / SDA - pin 3]
        SCL[GPIO3 / SCL - pin 5]
        GND[GND - pin 6]
        R[GPIO17 - pin 11]
        G[GPIO27 - pin 13]
        B[GPIO22 - pin 15]
    end

    subgraph OLED[SSD1306 OLED - I2C 0x3C]
        OV[VCC]
        OD[SDA]
        OC[SCL]
        OG[GND]
    end
    VCC --- OV
    SDA --- OD
    SCL --- OC
    GND --- OG

    R --- RR[330 ohm] --- RL[Red LED: anode + to cathode -]
    G --- GR[330 ohm] --- GL[Green LED: anode + to cathode -]
    B --- BR[330 ohm] --- BL[Blue LED: anode + to cathode -]
    RL --- GND
    GL --- GND
    BL --- GND
```

## Submission details

| Item | Location / details |
|---|---|
| GitHub | [markandrian30/MEX2-Voice-Controlled-Smart-Device](https://github.com/markandrian30/MEX2-Voice-Controlled-Smart-Device) - public - [MIT code licence](LICENSE) |
| Dataset | [Hugging Face source](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) - [Command manifest](data/command_manifest.csv) - [Full training manifest](data/full_manifest.csv). Source access terms apply; licence/DOI verification pending. |
| A100 cluster | DGX2 (`ai-n002`) - 1 x A100-SXM4, 40 GB (GPU 2) - seed 42 |
| Model weights | [Selected checkpoint](model/best_model.pt) - weights licence pending |

## 1. Model

![Model and example hardware](docs/Picture1.png)

- **Architecture:** A CRNN combines a CNN for audio feature extraction with a bidirectional GRU for learning patterns across the spoken command.
- **Audio input:** Variable-length mono audio at 16 kHz is converted to 64-band log-mel spectrograms, normalized per recording, using a 25 ms window and 10 ms hop.
- **CNN:** Three convolutional blocks use 3 x 3 kernels and 32, 64, and 128 filters. Each block applies LayerNorm, ReLU, and 2 x 2 max pooling.
- **Bidirectional GRU:** 64 hidden units per direction process 1,024 features per time step. The final forward and backward states form a 128-dimensional representation.
- **Classifier:** Dropout of 0.25 and a linear layer produce 33 class scores; softmax converts them to probabilities. The model has **515,937 trainable parameters**.
- **Command outputs:** 31 command classes cover 19 intents, with slot values encoded in the labels. Two additional classes recognize **Hello Kibo** and **Sagittarius**.
- **On-device prediction:** The model classifies a complete detected utterance on the Raspberry Pi. The live app requires a wake phrase before each command and accepts confidence **above 30%**.

## 2. Dataset

- **Main dataset:** [airimonda/ai231-me2-voice-commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands).
- **Preprocessing:** Matched command phrases and slot values, removed HF filenames already represented in GitHub, and reserved command holdout speakers.
- **Synthetic recordings:** GitHub synthetic recordings are included.

| Split | GitHub | HF | Total |
|---|---:|---:|---:|
| Training | 14,070 | 316 | 14,386 |
| Validation | 1,806 | 176 | 1,982 |
| Test | 1,614 | 56 | 1,670 |
| **Total** | **17,490** | **548** | **18,038** |

- **Dataset size:** 8.90 hours, 252 command speaker IDs, 19 intents, and 31 command classes.
- **Wake/exit recordings:** Another 1,044 recordings are training-only.
- **HF selection:** Of 615 eligible candidates, 67 were reserved for holdout.

**Command schema** - 13 fixed intents + 6 slotted intents; 3 phrase variations per command class.

| Intent | Phrase variations | Slot values |
|---|---|---|
| ALARM | Alarm {time}; Wake me up at {time}; Set an alarm for {time} | 6 AM, 8 AM, 9 PM |
| BRIGHTNESS | Brightness {percent}; Adjust brightness to {percent}; Brightness level {percent} | 100 percent, 20 percent, 60 percent |
| CALL | Call; Make a call; Make a phone call | - |
| COLOR | Change color to {color}; Switch color to {color}; Set color to {color} | blue, green, red |
| CREATE_REMINDER | Reminder {task}; Remind me to {task}; Create a reminder to {task} | drink water, exercise, study |
| LIGHT_OFF | Lights out; Kill the lights; Shut off the lights | - |
| LIGHT_ON | Lights on; Power on the lights; Turn on the lights | - |
| LIST_REMINDERS | Reminders; Show my reminders; List my reminders | - |
| MESSAGE | Message; Send a message; Send my message | - |
| NEXT | Next Song; Skip song; Play next song | - |
| PAUSE | Pause; Pause audio; Pause song | - |
| PLAY_MUSIC | Play Music; Start music; Play some music | - |
| STOP | Stop; Stop playing; End playback | - |
| TEMPERATURE | Temperature {degrees}; Change the temperature to {degrees}; Set the temperature to {degrees} | 18 degrees, 22 degrees, 26 degrees |
| TIME | Time; What time is it?; Tell me the time | - |
| TIMER | Timer {duration}; Countdown for {duration}; Start a timer for {duration} | 10 seconds, 1 minute, 30 seconds |
| VOLUME_DOWN | Volume down; Lower the volume; Turn the volume down | - |
| VOLUME_UP | Volume up; Increase the volume; Turn the volume up | - |
| WEATHER | Weather; What's the weather?; Tell me the weather | - |

**Controls:** wake: `Hello Kibo`; exit: `Sagittarius`. Slots are encoded in the command label.

[Complete schema (93 phrases)](data/command_schema.csv) - [Recorded phrase spellings](reports/training/phrases.csv). Matching normalizes case, punctuation and time notation (e.g. 8:00 AM = 8 AM).

## 3. Training on A100

| Item | Value |
|---|---|
| Cluster | 1 x A100-SXM4, 40 GB (GPU 2) |
| Objective / optimiser | Cross-entropy, label smoothing 0.05 / AdamW |
| Train / validation / test command speakers | 182 / 37 / 33 |
| Tuning | 16 trials; trial **2** selected by validation macro F1; seed 42 |
| GRU units per direction | Tested: 64, 128 -> **64** |
| Dropout | Tested: 0.25, 0.35 -> **0.25** |
| Learning rate | Tested: 0.001, 0.0005 -> initial **0.001**; checkpoint **0.0000625** |
| Weight decay | Tested: 0.0001, 0.001 -> **0.001** |
| LR decay / early stopping | Halve after 4 / stop after 10 epochs without validation F1 improvement |
| Selected checkpoint | **Epoch 57/60**; validation loss **0.2681** |
| GPU time | **~4.29 A100 GPU-hours**, all 16 trials; estimated from saved timestamps |

[Training records](reports/training) - [Timing calculation](reports/training/timing.json).

## 4. Validation on Raspberry Pi 5

**Pre-recorded audio validation**

| Item | Value |
|---|---|
| Command / intent accuracy | **83.66% / 84.65%** |
| False-accept rate | **18.75% (3/16)** |
| Inference p95 / mean RTF | **56.46 ms / 0.01144** |
| Runtime | PyTorch - 4 threads - Raspberry Pi 5 |

**Live voice validation**

| Item | Value |
|---|---|
| Command / intent accuracy | **71.56% / 71.56%** |
| False-accept rate | **31.25% (5/16)** |
| Inference p95 / mean RTF | **44.51 ms / 0.02176** |
| Runtime | PyTorch - 4 threads - Raspberry Pi 5 |

- Pre-recorded: **186 commands + 16 out-of-scope clips (202 total)**.
- Live: **93 commands + 16 out-of-scope clips (109 total)**, plus **16 no-wake trials**.

## Reviewer checklist

| Item | Status |
|---|---|
| Public repo + benchmark command | Included; live audio assets supplied separately |
| Dataset licence + DOI | Pending verification |
| Training logs + checkpoint | [Included](reports/training) |
| Pi timing | Measured on Pi 5; Pi 4 not tested |
| Unseen speakers | Bundled model: command split protected; auxiliary wake/exit speakers overlap holdout |
| Comparable-size baseline | Pending |
