import math
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

SR = 16000
N_FFT = 512
WIN = 400
HOP = 160
N_MELS = 64
FMIN = 50.0
FMAX = 7600.0

_MEL = {}

def hz_to_mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def mel_filterbank(device="cpu"):
    mmin, mmax = hz_to_mel(FMIN), hz_to_mel(FMAX)
    mels = np.linspace(mmin, mmax, N_MELS + 2)
    hz = np.array([mel_to_hz(m) for m in mels])
    bins = np.floor((N_FFT + 1) * hz / SR).astype(int)
    bins = np.clip(bins, 0, N_FFT // 2)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for i in range(1, N_MELS + 1):
        left, center, right = bins[i-1], bins[i], bins[i+1]
        if center <= left:
            center = min(left + 1, N_FFT // 2)
        if right <= center:
            right = min(center + 1, N_FFT // 2)
        for j in range(left, center):
            fb[i-1, j] = (j-left) / max(center-left, 1)
        for j in range(center, right):
            fb[i-1, j] = (right-j) / max(right-center, 1)
    return torch.tensor(fb, dtype=torch.float32, device=device)

def load_wav(path):
    """Load the complete recording; never crop/pad to a fixed duration."""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    wav = torch.as_tensor(wav, dtype=torch.float32).flatten()
    if sr != SR:
        new_n = max(1, round(len(wav) * SR / sr))
        wav = F.interpolate(wav[None, None, :], size=new_n,
                            mode="linear", align_corners=False)[0, 0]
    return wav

def wav_to_logmel(wav):
    """Natural-duration waveform -> variable-width log-mel spectrogram."""
    wav = torch.as_tensor(wav, dtype=torch.float32).flatten()
    if wav.numel() == 0:
        raise ValueError("Empty audio waveform")
    # torch.stft with center=True needs enough samples for reflection padding.
    min_samples = N_FFT // 2 + 1
    if wav.numel() < min_samples:
        wav = F.pad(wav, (0, min_samples - wav.numel()))

    window = torch.hann_window(WIN, device=wav.device)
    spec = torch.stft(
        wav, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        window=window, return_complex=True
    ).abs().pow(2)

    key = str(wav.device)
    if key not in _MEL:
        _MEL[key] = mel_filterbank(wav.device)
    mel = _MEL[key] @ spec
    feat = torch.log(mel + 1e-6)
    feat = (feat - feat.mean()) / (feat.std() + 1e-6)
    return feat.unsqueeze(0)
