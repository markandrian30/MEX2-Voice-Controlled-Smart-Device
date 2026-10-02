import argparse, sys, json
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
from runtime import load_model, load_wav, wav_to_logmel, accept

p=argparse.ArgumentParser()
p.add_argument('wav', type=Path)
p.add_argument('--min-confidence', type=float, default=.25)
a=p.parse_args()
if not 0 <= a.min_confidence <= 1: p.error('Threshold must be in [0,1]')
torch.set_num_threads(4)
model,cp=load_model(Path(__file__).resolve().parent/'model/best_model.pt')
with torch.inference_mode():
    probabilities=model(wav_to_logmel(load_wav(a.wav)).unsqueeze(0)).softmax(1)[0]
    raw=cp['labels'][int(probabilities.argmax())]; confidence=float(probabilities.max())
print(json.dumps(dict(prediction=accept(raw,confidence,a.min_confidence),raw_prediction=raw,confidence=confidence)))
