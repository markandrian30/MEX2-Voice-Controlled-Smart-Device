from pathlib import Path
import torch
from crnn_model import VoiceCRNN, MODEL_VERSION
from features import load_wav, wav_to_logmel, SR

def load_model(path):
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    if checkpoint['model_version'] != MODEL_VERSION:
        raise ValueError('Unsupported model version')
    model = VoiceCRNN(len(checkpoint['labels']), checkpoint.get('dropout', .35), checkpoint['hidden_size'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval(), checkpoint

def accept(label, confidence, threshold):
    return label if confidence > threshold else 'OUT_OF_SCOPE'

def intent(label):
    for prefix in ('CREATE_REMINDER', 'TEMPERATURE', 'BRIGHTNESS', 'ALARM', 'TIMER', 'COLOR'):
        if label.startswith(prefix + '_'):
            return prefix
    return label
