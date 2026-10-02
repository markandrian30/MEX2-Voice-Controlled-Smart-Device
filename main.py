#!/usr/bin/env python3
"""Wake once at startup; wake for music commands; Sagittarius exits after activation."""
import argparse
from collections import deque
from contextlib import contextmanager
import os
from pathlib import Path
import queue
import time
import threading
import wave

import sys

import numpy as np
import torch
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


import torch
import torch.nn as nn

MODEL_VERSION = "voicecnn_length_aware_v2"

WAKE_PHRASES = {'HELLO_KIBO': 'Hello Kibo'}
EXIT_LABEL = 'SAGITTARIUS'
EXIT_PHRASE = 'Sagittarius'
WAKE_PROMPT = 'HELLO KIBO'


def time_mask(x, lengths):
    valid = torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None]
    return valid[:, None, None, :]


class LengthAwareBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        # Normalize channels at each frequency/time position, never across
        # recordings or time. Padding cannot affect valid-frame statistics.
        self.norm = nn.LayerNorm(out_channels)
        self.pool = nn.MaxPool2d(2, ceil_mode=True) if pool else None

    def forward(self, x, lengths):
        x = x.masked_fill(~time_mask(x, lengths), 0)
        x = self.conv(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = torch.relu(x)
        # Remove invalid responses before they enter another receptive field.
        x = x.masked_fill(~time_mask(x, lengths), 0)
        if self.pool is not None:
            # Keep the last valid frame of odd-length clips. After ReLU,
            # invalid zeros cannot win a partially valid max-pooling window.
            x = self.pool(x)
            lengths = (lengths + 1) // 2
            x = x.masked_fill(~time_mask(x, lengths), 0)
        return x, lengths


class VoiceCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.35):
        super().__init__()
        self.blocks = nn.ModuleList([
            LengthAwareBlock(1, 32),
            LengthAwareBlock(32, 64),
            LengthAwareBlock(64, 128),
            LengthAwareBlock(128, 160, pool=False),
        ])
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(160, num_classes))

    def forward(self, x, lengths=None):
        if x.ndim != 4 or x.shape[-1] < 1:
            raise ValueError("Expected nonempty [batch, channel, mel, time] features")
        if lengths is None:
            # Single-recording inference has no artificial batch padding.
            lengths = torch.full((x.shape[0],), x.shape[-1], dtype=torch.long, device=x.device)
        else:
            lengths = torch.as_tensor(lengths, dtype=torch.long, device=x.device)
            if lengths.shape != (x.shape[0],):
                raise ValueError("Supply one feature-frame length per recording")
            if bool(((lengths < 1) | (lengths > x.shape[-1])).any()):
                raise ValueError("Feature lengths must be between 1 and the padded width")
        for block in self.blocks:
            x, lengths = block(x, lengths)
        denominator = (lengths * x.shape[-2]).to(x.dtype).unsqueeze(1)
        embedding = x.sum(dim=(2, 3)) / denominator
        return self.classifier(embedding)

import random
import subprocess


MUSIC_COMMANDS = {'PLAY_MUSIC', 'PAUSE', 'STOP', 'NEXT', 'VOLUME_UP', 'VOLUME_DOWN'}


LED_COMMANDS = {'COLOR_RED', 'LIGHT_OFF', 'COLOR_BLUE', 'BRIGHTNESS_100', 'COLOR_GREEN', 'BRIGHTNESS_20', 'LIGHT_ON', 'BRIGHTNESS_60'}


class LEDController:
    """Four separate active-high LEDs; BCM numbering, PWM brightness."""
    PINS = {'RED': 17, 'BLUE': 22, 'GREEN': 27}

    def __init__(self, pin_factory=None):
        self.leds = {}
        self._owns_factory = pin_factory is None
        self._factory = pin_factory
        self.brightness = 1.0
        self.brightness_percent = 100
        self.selected = set(self.PINS)
        self.enabled = False
        # Keep microphone inference usable on laptops and DGX servers.
        if pin_factory is None:
            model_path = Path('/proc/device-tree/model')
            if not model_path.is_file() or b'Raspberry Pi' not in model_path.read_bytes():
                return
        from gpiozero import PWMLED
        try:
            if self._factory is None:
                from gpiozero.pins.lgpio import LGPIOFactory
                self._factory = LGPIOFactory()
            for color, pin in self.PINS.items():
                self.leds[color] = PWMLED(pin, active_high=True, initial_value=0,
                                          frequency=100, pin_factory=self._factory)
        except BaseException:
            self.close()
            raise

    def handle(self, label):
        if label not in LED_COMMANDS:
            return None
        if not self.leds:
            raise RuntimeError('GPIO LED control is available only on a configured Raspberry Pi.')
        if label == 'LIGHT_OFF':
            self.enabled = False
            message = 'Lights off.'
        elif label == 'LIGHT_ON':
            self.selected = set(self.PINS)
            self.enabled = True
            message = f'All LEDs on at {self.brightness_percent}%.'
        elif label.startswith('COLOR_'):
            color = label.removeprefix('COLOR_')
            self.selected = {color}
            self.enabled = True
            message = f'{color.title()} LED only at {self.brightness_percent}%.'
        else:
            self.brightness = {'BRIGHTNESS_20': 0.05, 'BRIGHTNESS_60': 0.50, 'BRIGHTNESS_100': 1.0}[label]
            self.brightness_percent = {'BRIGHTNESS_20': 20, 'BRIGHTNESS_60': 60, 'BRIGHTNESS_100': 100}[label]
            self.enabled = True
            message = f'LED brightness: {self.brightness_percent}%.'
        # Turn unselected LEDs off before lighting the selected color.
        for color, led in self.leds.items():
            if not self.enabled or color not in self.selected:
                led.value = 0
        for color, led in self.leds.items():
            if self.enabled and color in self.selected:
                led.value = self.brightness
        return message

    def close(self):
        try:
            for led in self.leds.values():
                try:
                    led.off()
                finally:
                    led.close()
        finally:
            self.leds.clear()
            if self._owns_factory and self._factory is not None:
                self._factory.close()
                self._factory = None


class MicrophoneGate:
    """Discard PAUSE/STOP output tails and wait for quiet before rearming."""
    def __init__(self, quiet_seconds=.20):
        self.quiet_samples = max(1, int(quiet_seconds * SR))
        self.after = float('-inf')
        self.quiet = 0
        self.armed = True

    def reset(self, settle=.25, now=None):
        self.after = (time.monotonic() if now is None else now) + settle
        self.quiet = 0
        self.armed = False

    def ready(self, captured, rms, samples, threshold):
        if captured <= self.after:
            return False
        if self.armed:
            return True
        self.quiet = self.quiet + samples if rms < threshold else 0
        if self.quiet >= self.quiet_samples:
            self.armed = True
        return False


class MusicCommandGate:
    """Wake once at startup, then before each command while music plays."""
    def __init__(self, player, timeout=30.):
        self.player = player
        self.timeout = timeout
        self.deadline = None
        self.activated = False

    def refresh(self, now):
        if self.deadline is not None and (
                self.player.state != 'playing' or now >= self.deadline):
            self.complete()
            return True
        return False

    def accepts(self, now):
        self.refresh(now)
        return self.activated and (self.player.state != 'playing' or self.deadline is not None)

    def wake(self, now):
        self.refresh(now)
        self.activated = True
        if self.player.state == 'playing':
            self.deadline = now + self.timeout
            self.player.set_ducked(True)
            return True
        return False

    def complete(self):
        self.deadline = None
        self.player.set_ducked(False)


class MusicPlayer:
    rate = 48000

    def __init__(self, directory, volume=50):
        self.directory = Path(directory).expanduser()
        self.volume = max(0, min(100, int(volume)))
        self.state = 'stopped'
        self.track = None
        self.position = 0
        self.audio = None
        self._lock = threading.Lock()
        self._held = False
        self._ducked = False
        self._response_mix = False
        self._response = None
        self._response_position = 0
        self._response_done = threading.Event()
        self._stream = None
        self.underflows = 0
        self._advance = threading.Event()
        self._closed = False
        self._revision = 0
        self._advance_worker = None

    def start(self, sd):
        self._stream = sd.OutputStream(
            samplerate=self.rate, channels=2, dtype='float32',
            blocksize=960, latency='high', callback=self.render)
        try:
            self._stream.start()
            self._advance_worker = threading.Thread(target=self._advance_loop,
                                                     name='music-next', daemon=True)
            self._advance_worker.start()
        except BaseException:
            self._stream.close()
            self._stream = None
            raise

    def close(self):
        with self._lock:
            self._closed = True
            self._revision += 1
        self._advance.set()
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None
        self._response_done.set()

    def render(self, outdata, frames, time_info, status):
        """No file I/O or decoding in the audio callback."""
        outdata.fill(0)
        if status:
            self.underflows += 1
        if not self._lock.acquire(blocking=False):
            return
        try:
            if not self._held and self.state == 'playing' and self.audio is not None:
                n = min(frames, len(self.audio) - self.position)
                if self._response_mix or self._response is not None:
                    volume = min(self.volume, 10)
                elif self._ducked:
                    volume = min(self.volume, 5)
                else:
                    volume = self.volume
                np.multiply(self.audio[self.position:self.position + n],
                            volume / 100., out=outdata[:n])
                self.position += n
                if n and self.position == len(self.audio):
                    self._advance.set()
            if self._response is not None:
                n = min(frames, len(self._response) - self._response_position)
                voice = self._response[self._response_position:self._response_position + n]
                # Preserve normal speech gain; trim only music peaks if the mix
                # would exceed the speaker's [-1, 1] sample range.
                np.clip(outdata[:n], -1. - voice, 1. - voice, out=outdata[:n])
                outdata[:n] += voice
                self._response_position += n
                if self._response_position == len(self._response):
                    self._response = None
                    self._response_done.set()
        finally:
            self._lock.release()

    def set_ducked(self, enabled):
        # Preserve the user's volume separately so VOLUME_UP/DOWN still work.
        with self._lock:
            self._ducked = bool(enabled)

    @contextmanager
    def interruption(self, hold=True):
        """Hold music for controls, or keep it playing softly under speech."""
        with self._lock:
            previous_held, previous_ducked = self._held, self._ducked
            previous_mix = self._response_mix
            self._held = hold
            if not hold:
                self._ducked = True
                self._response_mix = True
        try:
            yield
        finally:
            with self._lock:
                self._held, self._ducked = previous_held, previous_ducked
                self._response_mix = previous_mix

    def say(self, audio, rate):
        audio = np.asarray(audio, dtype=np.float32)
        if rate != self.rate:
            length = max(1, round(len(audio) * self.rate / rate))
            audio = np.interp(np.arange(length) * rate / self.rate,
                              np.arange(len(audio)), audio).astype(np.float32)
        stereo = np.repeat(audio[:, None], 2, axis=1)
        if self._stream is None or not self._stream.active:
            raise RuntimeError('Speaker stream is not running.')
        with self._lock:
            self._response_done.clear()
            self._response_position = 0
            self._response = stereo
        deadline = time.monotonic() + len(stereo) / self.rate + 5
        while not self._response_done.wait(.1):
            if not self._stream.active or time.monotonic() > deadline:
                with self._lock:
                    self._response = None
                raise RuntimeError('Speaker stopped while playing a response.')
        # The last submitted buffer must reach the speaker before listening resumes.
        time.sleep(float(self._stream.latency))

    def _decode(self, path):
        try:
            result = subprocess.run(
                ['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path),
                 '-f', 'f32le', '-ac', '2', '-ar', str(self.rate), 'pipe:1'],
                capture_output=True, timeout=60, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError('Music playback requires ffmpeg.') from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'Cannot decode {path.name}: {exc.stderr.decode(errors="replace").strip()}') from exc
        audio = np.frombuffer(result.stdout, dtype='<f4').reshape(-1, 2)
        if not len(audio):
            raise RuntimeError(f'No audio in {path.name}.')
        return audio

    def _advance_loop(self):
        while True:
            self._advance.wait()
            self._advance.clear()
            with self._lock:
                if self._closed:
                    return
                revision = self._revision
            try:
                if self._choose_track('playing', sequential=True, automatic=True):
                    print(f'Next: {self.track.name} (playing)', flush=True)
            except Exception as exc:
                with self._lock:
                    if self._revision != revision or self._closed:
                        continue
                    self.state = 'stopped'
                print(f'Could not play next song: {exc}', flush=True)

    def _choose_track(self, state, sequential=False, automatic=False):
        with self._lock:
            if self._closed:
                return False
            if automatic:
                if self.state != 'playing' or self.audio is None or self.position < len(self.audio):
                    return False
            else:
                self._revision += 1
            revision = self._revision
            previous = self.track
        tracks = sorted(p for p in self.directory.iterdir()
                        if p.is_file() and p.suffix.lower() == '.mp3') if self.directory.is_dir() else []
        if not tracks:
            raise RuntimeError(f'No MP3 files found in {self.directory}.')
        if sequential:
            index = (tracks.index(previous) + 1) % len(tracks) if previous in tracks else 0
            chosen = tracks[index]
        else:
            choices = [p for p in tracks if p != previous] or tracks
            chosen = random.choice(choices)
        audio = self._decode(chosen)
        with self._lock:
            if self._closed or revision != self._revision:
                return False
            if automatic and self.state != 'playing':
                return False
            self.track = chosen
            self.audio = audio
            self.position = 0
            self.state = state
            return True

    def handle(self, label):
        label = label.upper()
        if label not in MUSIC_COMMANDS:
            return None
        if label == 'PLAY_MUSIC':
            with self._lock:
                if self.state == 'paused':
                    self.state = 'playing'
                    if self.audio is not None and self.position >= len(self.audio):
                        self._advance.set()
                    return f'Resuming: {self.track.name}'
                if self.state == 'playing':
                    return f'Already playing: {self.track.name}'
            self._choose_track('playing')
            return f'Playing: {self.track.name}'
        if label == 'NEXT':
            # Neither NEXT nor volume changes may resume paused/stopped music.
            self._choose_track(self.state, sequential=True)
            return f'Next: {self.track.name} ({self.state})'
        with self._lock:
            if label == 'PAUSE':
                self._revision += 1
                self._advance.clear()
                if self.state == 'playing':
                    self.state = 'paused'
                return 'Music paused.' if self.state == 'paused' else 'Music is stopped.'
            if label == 'STOP':
                self._revision += 1
                self._advance.clear()
                self.state = 'stopped'
                self.position = 0
                self.track = None
                self.audio = None
                return 'Music stopped.'
            self.volume = max(0, min(100, self.volume + (25 if label == 'VOLUME_UP' else -25)))
            return f'Music volume: {self.volume}%'


@contextmanager
def single_instance():
    """Avoid independent players competing for the same voice commands."""
    if os.name != 'posix':
        yield
        return
    import fcntl
    lock_dir = Path.home() / '.cache' / 'mex2'
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / 'voice.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Voice program is already running. Stop its terminal with Ctrl+C first.')
        yield


from torch.nn.utils.rnn import pack_padded_sequence


class VoiceCRNN(nn.Module):
    def __init__(self, num_classes, dropout=0.35, hidden_size=64):
        super().__init__()
        self.blocks = nn.ModuleList([
            LengthAwareBlock(1, 32), LengthAwareBlock(32, 64),
            LengthAwareBlock(64, 128),
        ])
        self.gru = nn.GRU(128 * 8, hidden_size, batch_first=True,
                          bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(hidden_size * 2, num_classes))

    def forward(self, x, lengths=None):
        if x.ndim != 4 or x.shape[1:3] != (1, 64) or x.shape[-1] < 1:
            raise ValueError("Expected nonempty [batch, 1, 64, time] features")
        if lengths is None:
            lengths = torch.full((x.shape[0],), x.shape[-1], device=x.device,
                                 dtype=torch.long)
        else:
            lengths = torch.as_tensor(lengths, dtype=torch.long, device=x.device)
        if lengths.shape != (x.shape[0],) or bool(((lengths < 1) | (lengths > x.shape[-1])).any()):
            raise ValueError("Supply one valid feature-frame length per recording")
        for block in self.blocks:
            x, lengths = block(x, lengths)
        sequence = x.permute(0, 3, 1, 2).flatten(2)
        packed = pack_padded_sequence(sequence, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.classifier(torch.cat((hidden[-2], hidden[-1]), dim=1))


def load_model(path):
    if not path.is_file():
        raise SystemExit(f'Model missing: {path}. Pass --model /path/to/best_model.pt.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    is_crnn = checkpoint.get('model_version') == 'voicecrnn_length_aware_v1'
    if checkpoint.get('model_version') != MODEL_VERSION and not is_crnn:
        raise SystemExit(f'Incompatible model version: {path}')
    missing = (set(WAKE_PHRASES) | {EXIT_LABEL}) - set(checkpoint.get('labels', []))
    if missing:
        raise SystemExit(f'Model missing wake/exit classes: {", ".join(sorted(missing))}. Run fresh training/tuning.')
    if set(checkpoint.get('wake_labels', [])) != set(WAKE_PHRASES):
        raise SystemExit('This checkpoint does not use HELLO_KIBO as its only wake word. Run fresh training/tuning.')
    if checkpoint.get('exit_label') != EXIT_LABEL or 'OKAY_THANK_YOU' in checkpoint['labels']:
        raise SystemExit('Checkpoint must use SAGITTARIUS to exit and exclude OKAY_THANK_YOU. Run fresh training/tuning.')
    if is_crnn:
        model = VoiceCRNN(len(checkpoint['labels']), dropout=checkpoint.get('dropout', .35),
                          hidden_size=checkpoint['hidden_size'])
    else:
        model = VoiceCNN(len(checkpoint['labels']), dropout=checkpoint.get('dropout', .35))
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval(), checkpoint


def default_paths(script_dir):
    bundled = script_dir / 'model' / 'best_model.pt'
    if bundled.is_file():
        return bundled
    # Support MEX2/train and copied result-folder demos.
    root = next((p for p in script_dir.parents if (p / 'train').is_dir() and any((p / name).is_dir() for name in ('results', 'results2'))), script_dir.parent)
    command = script_dir / 'best_model.pt'
    if not command.exists():
        command = root / 'results' / 'best_model.pt'
    if not command.exists():
        candidates = [p for p in (root / 'results').glob('*/best_model.pt')
                      if (p.parent / 'summary.json').is_file()]
        if candidates:
            command = max(candidates, key=lambda p: p.stat().st_mtime)
    return command


RESPONSE_FILES = {'ALARM_6_00AM': 'ALARM_6_00AM/setting_alarm_for_6_am.wav', 'ALARM_4_00AM': 'ALARM_4_00AM/setting_alarm_for_4_am.wav', 'WAKE': 'WAKE/hi_there.wav', 'PLAY_MUSIC': 'PLAY_MUSIC/playing_music.wav', 'VOLUME_UP': 'VOLUME_UP/turning_volume_up.wav', 'VOLUME_DOWN': 'VOLUME_DOWN/turning_volume_down.wav', 'NEXT': 'NEXT/playing_next_song.wav', 'PAUSE': 'PAUSE/music_paused.wav', 'STOP': 'STOP/music_stopped.wav', 'LIGHT_ON': 'LIGHT_ON/turning_lights_on.wav', 'LIGHT_OFF': 'LIGHT_OFF/turning_lights_off.wav', 'BRIGHTNESS_20': 'BRIGHTNESS_20/setting_brightness_to_20_percent.wav', 'BRIGHTNESS_60': 'BRIGHTNESS_60/setting_brightness_to_60_percent.wav', 'BRIGHTNESS_100': 'BRIGHTNESS_100/setting_brightness_to_100_percent.wav', 'COLOR_RED': 'COLOR_RED/changing_lights_to_red.wav', 'COLOR_BLUE': 'COLOR_BLUE/changing_lights_to_blue.wav', 'TEMPERATURE_18': 'TEMPERATURE_18/setting_temperature_to_18_degrees.wav', 'TEMPERATURE_22': 'TEMPERATURE_22/setting_temperature_to_22_degrees.wav', 'TEMPERATURE_26': 'TEMPERATURE_26/setting_temperature_to_26_degrees.wav', 'WEATHER': 'WEATHER/checking_the_weather.wav', 'TIMER_10S': 'TIMER_10S/setting_timer_for_10_seconds.wav', 'TIMER_30S': 'TIMER_30S/setting_timer_for_30_seconds.wav', 'TIMER_1M': 'TIMER_1M/setting_timer_for_1_minute.wav', 'ALARM_5_00AM': 'ALARM_5_00AM/setting_alarm_for_5_am.wav', 'ALARM_8_00AM': 'ALARM_8_00AM/setting_alarm_for_8_am.wav', 'ALARM_9_00PM': 'ALARM_9_00PM/setting_alarm_for_9_pm.wav', 'CALL': 'CALL/making_the_call.wav', 'MESSAGE': 'MESSAGE/sending_the_message.wav', 'CREATE_REMINDER_DRINK_WATER': 'CREATE_REMINDER_DRINK_WATER/remind_you_to_drink_water.wav', 'CREATE_REMINDER_STUDY': 'CREATE_REMINDER_STUDY/remind_you_to_study.wav', 'CREATE_REMINDER_EXERCISE': 'CREATE_REMINDER_EXERCISE/remind_you_to_exercise.wav', 'LIST_REMINDERS': 'LIST_REMINDERS/here_are_your_reminders.wav', 'COLOR_GREEN': 'COLOR_GREEN/changing_lights_to_green.wav'}


OPTIONAL_RESPONSE_LABELS = {'ALARM_5_00AM'}


def response_available(label, path):
    return label not in OPTIONAL_RESPONSE_LABELS or path.is_file()


def responses_path(script_dir):
    for parent in (script_dir, *script_dir.parents):
        for candidate in (parent / 'responses', parent / 'results' / 'responses'):
            if candidate.is_dir():
                return candidate
    return script_dir.parent / 'responses'


def read_response(path):
    if not path.is_file():
        raise SystemExit(f'Male-voice response missing: {path}')
    with wave.open(str(path), 'rb') as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise SystemExit('Activation response must be mono PCM16 WAV.')
        rate = wav.getframerate()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768.
    if not len(audio):
        raise SystemExit('Activation response is empty.')
    return audio, rate


def spoken_time_text(now=None):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo('Asia/Manila')) if now is None else now
    hour = now.hour % 12 or 12
    minute = "o'clock" if now.minute == 0 else (f'oh {now.minute}' if now.minute < 10 else str(now.minute))
    period = 'A M' if now.hour < 12 else 'P M'
    return f'The time is {hour} {minute} {period}.'



def reminder_response_files(reminders):
    """Use the reference speaker's numbered clips in saved reminder order."""
    if not reminders:
        return ['LIST_REMINDERS/no_reminders.wav']
    names = {'Study': 'study', 'Exercise': 'exercise', 'Drink water': 'drink_water'}
    return ['LIST_REMINDERS/here_are_your_reminders.wav'] + [
        f'LIST_REMINDERS/{index}_{names[item]}.wav'
        for index, item in enumerate(reminders, 1)]


def synthesize_response_audio(text):
    """Offline speech returned as PCM in memory; never writes a response WAV."""
    import io
    import shutil
    engine = shutil.which('espeak-ng')
    env = os.environ.copy()
    extra = []
    if engine is None:
        root = Path.home() / '.local/share/mex2-time-tts/root'
        engine = root / 'usr/bin/espeak-ng'
        data = next(iter(root.glob('usr/lib/*/espeak-ng-data')), None)
        if not engine.is_file() or data is None:
            raise RuntimeError('Offline speech engine is not installed.')
        env['LD_LIBRARY_PATH'] = str(data.parent) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        extra = ['--path=' + str(data.parent)]
    result = subprocess.run([str(engine), *extra, '--stdout', '-v', 'en-us', '-s', '155', text],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=10)
    if result.returncode:
        raise RuntimeError('Speech synthesis failed: ' + result.stderr.decode(errors='replace').strip())
    with wave.open(io.BytesIO(result.stdout), 'rb') as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise RuntimeError('Unexpected speech audio format.')
        rate = wav.getframerate()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768.
    if not len(audio):
        raise RuntimeError('Speech synthesis returned empty audio.')
    return audio, rate



def timer_alarm_audio(rate=48000):
    """Three short beeps, generated locally without an audio file or internet."""
    samples = int(.35 * rate)
    t = np.arange(samples, dtype=np.float32) / rate
    envelope = np.minimum(1., np.minimum(t / .015, (.35 - t) / .015))
    tone = (.35 * np.sin(2 * np.pi * 880 * t) * envelope).astype(np.float32)
    gap = np.zeros(int(.2 * rate), dtype=np.float32)
    return np.concatenate([tone, gap, tone, gap, tone]), rate


class WeatherDisplay:
    """OLED weather, Philippine time, and up to three session reminders."""
    def __init__(self):
        self.device = None
        self.lock = threading.Lock()
        self.closed = False
        self.worker = None
        self.cached = None
        self.cached_at = 0.
        self.generation = 0
        self.reminders = []
        self.music_screen = False
        self.music_key = None
        self.music_checked = 0.
        self.music_started = 0.
        self.timer_cancel = threading.Event()
        self.timer_worker = None
        self.on_timer_done = None
        self.on_weather_ready = None
        self.display_event = threading.Event()
        self.display_pending = None
        self.display_worker = threading.Thread(target=self._display_loop,
                                               name='oled-display', daemon=True)
        self.display_worker.start()

    @staticmethod
    def fetch():
        import json
        import urllib.request
        url = ('https://api.open-meteo.com/v1/forecast?latitude=14.6488&longitude=121.0509'
               '&current=temperature_2m,relative_humidity_2m,weather_code'
               '&temperature_unit=celsius&timezone=Asia%2FManila&forecast_days=1')
        with urllib.request.urlopen(url, timeout=8) as response:
            current = json.loads(response.read(65536))['current']
        temperature = float(current['temperature_2m'])
        humidity = float(current['relative_humidity_2m'])
        if not math.isfinite(temperature) or not math.isfinite(humidity) or not 0 <= humidity <= 100:
            raise ValueError('Invalid weather data')
        code = int(current['weather_code'])
        conditions = {0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
                      45: 'Fog', 48: 'Freezing fog', 51: 'Light drizzle', 53: 'Drizzle',
                      55: 'Heavy drizzle', 56: 'Freezing drizzle', 57: 'Freezing drizzle',
                      61: 'Light rain', 63: 'Rain', 65: 'Heavy rain', 66: 'Freezing rain',
                      67: 'Freezing rain', 71: 'Light snow', 73: 'Snow', 75: 'Heavy snow',
                      77: 'Snow grains', 80: 'Light showers', 81: 'Rain showers',
                      82: 'Heavy showers', 85: 'Snow showers', 86: 'Heavy snow showers',
                      95: 'Thunderstorm', 96: 'Storm with hail', 99: 'Storm with hail'}
        return [f'{temperature:.1f} C  RH {humidity:.0f}%',
                conditions.get(code, 'Conditions unknown'),
                'As of ' + str(current['time']).split('T')[-1][:5] + ' PHT']

    def draw(self, lines, generation=None, scroll_elapsed=0.):
        # Coalesce requests; no imports, rendering, or I2C on the audio thread.
        with self.lock:
            if self.closed or (generation is not None and generation != self.generation):
                return
            self.display_pending = (tuple(lines), self.generation)
            self.display_event.set()

    def _display_loop(self):
        current = None
        scrolling = False
        started = 0.
        self.display_visible = False
        try:
            while True:
                self.display_event.wait(.15 if scrolling else None)
                with self.lock:
                    if self.closed:
                        break
                    pending = self.display_pending
                    self.display_pending = None
                    self.display_event.clear()
                    generation = self.generation
                if pending is not None:
                    lines, token = pending
                    if token != generation:
                        continue
                    if lines is None:
                        current = None
                        scrolling = False
                        try:
                            if self.device is not None:
                                self.device.hide()
                                self.display_visible = False
                        except Exception as exc:
                            print(f'OLED unavailable: {exc}', flush=True)
                        continue
                    if current is None or current[0][:3] != lines[:3]:
                        started = time.monotonic()
                    current = (lines, token)
                if current is None or current[1] != generation:
                    scrolling = False
                    continue
                try:
                    # The hardware is owned exclusively by this worker.
                    scrolling = self._render_lines(current[0], time.monotonic() - started)
                except Exception as exc:
                    print(f'OLED unavailable: {exc}', flush=True)
                    current = None
                    scrolling = False
        finally:
            if self.device is not None:
                try:
                    self.device.cleanup()
                except Exception as exc:
                    print(f'OLED cleanup: {exc}', flush=True)

    def _render_lines(self, lines, scroll_elapsed):
        if self.device is None:
            from luma.core.interface.serial import i2c
            from luma.oled.device import ssd1306
            from PIL import ImageFont
            self.device = ssd1306(i2c(port=1, address=0x3C), width=128, height=64)
            self.font = ImageFont.load_default(size=10)
        if not self.display_visible:
            self.device.show()
            self.display_visible = True
        from luma.core.render import canvas
        scrolling = False
        with canvas(self.device) as draw:
            for index, line in enumerate(lines):
                if lines[0] in {'Now playing...', 'Paused'} and index == 2:
                    width = draw.textlength(line, font=self.font)
                    if width > 126:
                        scrolling = True
                        # Pause briefly, then loop the complete title leftwards.
                        offset = int(max(0., scroll_elapsed - 1.) * 24) % (math.ceil(width) + 32)
                        draw.text((1 - offset, 20), line, font=self.font, fill='white')
                        draw.text((1 - offset + math.ceil(width) + 32, 20),
                                  line, font=self.font, fill='white')
                        continue
                while line and draw.textbbox((0, 0), line, font=self.font)[2] > 126:
                    line = line[:-1]
                draw.text((1, index * 10), line, font=self.font, fill='white')
        return scrolling

    def handle(self, label):
        # NEXT updates the existing music screen without blanking the OLED.
        if not (label == 'NEXT' and self.music_screen):
            self.dismiss()
        self.music_screen = label in MUSIC_COMMANDS
        reminders = {
            'CREATE_REMINDER_STUDY': 'Study',
            'CREATE_REMINDER_EXERCISE': 'Exercise',
            'CREATE_REMINDER_DRINK_WATER': 'Drink water',
        }
        if label == 'WEATHER':
            self.request()
            return
        if label in reminders:
            item = reminders[label]
            added = item not in self.reminders
            if added:
                self.reminders.append(item)
            lines = ['Reminder added' if added else 'Already on your list', item,
                     '', f'{len(self.reminders)}/3 reminders']
        elif label == 'LIST_REMINDERS':
            lines = [f'Reminders ({len(self.reminders)}/3)', '']
            lines += ([f'{i}. {item}' for i, item in enumerate(self.reminders, 1)]
                      or ['No reminders yet'])
        elif label in {'ALARM_4_00AM', 'ALARM_5_00AM', 'ALARM_6_00AM', 'ALARM_8_00AM', 'ALARM_9_00PM'}:
            alarm_time = {'ALARM_6_00AM': '6:00 AM', 'ALARM_4_00AM': '4:00 AM', 'ALARM_5_00AM': '5:00 AM', 'ALARM_8_00AM': '8:00 AM',
                          'ALARM_9_00PM': '9:00 PM'}[label]
            lines = ['Alarm', '', alarm_time, '', 'Display only', 'Not scheduled']
        elif label in {'TEMPERATURE_18', 'TEMPERATURE_22', 'TEMPERATURE_26'}:
            degrees = label.rsplit('_', 1)[1]
            lines = ['Temperature', '', degrees + ' C']
        elif label in {'TIMER_10S', 'TIMER_30S', 'TIMER_1M'}:
            seconds = {'TIMER_10S': 10, 'TIMER_30S': 30, 'TIMER_1M': 60}[label]
            self.start_timer(seconds)
            duration = '1 minute' if seconds == 60 else f'{seconds} seconds'
            return f'Starting a timer for {duration}.'
        elif label == 'TIME':
            from datetime import datetime
            from zoneinfo import ZoneInfo
            now = datetime.now(ZoneInfo('Asia/Manila'))
            lines = ['Quezon City', '', now.strftime('%I:%M %p').lstrip('0'),
                     now.strftime('%a, %d %b %Y'), 'Philippine time']
        else:
            return
        if label in reminders:
            response_text = f"I'll remind you to {item.lower()}."
        elif label == 'LIST_REMINDERS':
            response_text = ('Your reminders: ' + ', '.join(item.lower() for item in self.reminders) + '.'
                             if self.reminders else 'You have no reminders yet.')
        elif label.startswith('ALARM_'):
            response_text = 'Setting the alarm for ' + alarm_time.replace(':00', '') + '.'
        elif label.startswith('TEMPERATURE_'):
            response_text = f'Setting the temperature to {degrees} degrees.'
        else:
            response_text = None  # TIME prints its spoken answer in play_response.
        try:
            self.draw(lines, self.generation)
        except Exception as exc:
            print(f'OLED unavailable: {exc}', flush=True)

        return response_text

    def refresh_music(self, player, force=False):
        if not self.music_screen or self.closed:
            return
        now = time.monotonic()
        if not force and now - self.music_checked < .1:
            return
        self.music_checked = now
        with player._lock:
            key = (player.state, player.track, player.volume)
        if key == self.music_key:
            return
        if self.music_key is None or key[:2] != self.music_key[:2]:
            self.music_started = now
        self.music_key = key
        state, track, volume = key
        if state not in {'playing', 'paused'} or track is None:
            self.dismiss()
            return
        title = ' '.join(track.stem.replace('_', ' ').split()) or 'Unknown title'
        lines = ['Paused' if state == 'paused' else 'Now playing...', '', title, '', '', f'Volume: {volume}%']
        try:
            self.draw(lines, self.generation, scroll_elapsed=now - self.music_started)
        except Exception as exc:
            print(f'OLED unavailable: {exc}', flush=True)

    def start_timer(self, seconds):
        self.timer_started_at = None
        with self.lock:
            if self.closed:
                return
            self.timer_cancel.set()
            self.timer_cancel = threading.Event()
            self.timer_started_at = time.monotonic()
            self.timer_worker = threading.Thread(
                target=self._countdown,
                args=(self.timer_started_at + seconds, self.generation, self.timer_cancel),
                name='oled-timer', daemon=True)
            self.timer_worker.start()

    def _countdown(self, deadline, generation, cancel):
        previous = None
        while not cancel.is_set():
            remaining = max(0, math.ceil(deadline - time.monotonic()))
            if remaining != previous:
                minutes, seconds = divmod(remaining, 60)
                try:
                    self.draw(['Timer', '', f'{minutes:02d}:{seconds:02d}',
                               '', 'Time is up!' if remaining == 0 else 'Counting down...'], generation)
                except Exception as exc:
                    print(f'OLED unavailable: {exc}', flush=True)
                previous = remaining
            if remaining == 0:
                if not cancel.is_set():
                    print('\nTimer finished.\n', flush=True)
                    if self.on_timer_done is not None:
                        self.on_timer_done(cancel)
                return
            cancel.wait(min(.2, max(.01, deadline - time.monotonic())))

    def dismiss(self):
        with self.lock:
            self.music_screen = False
            self.music_key = None
            # Other commands hide the countdown; the timer and alarm continue.
            # Invalidate any outstanding API response before blanking the OLED.
            self.generation += 1
            if not self.closed:
                self.display_pending = (None, self.generation)
                self.display_event.set()

    def request(self):
        with self.lock:
            if self.closed:
                return
            self.generation += 1
            generation = self.generation
            self.worker = threading.Thread(target=self._update, args=(generation,),
                                           name='weather-oled', daemon=True)
            self.worker.start()

    def _update(self, generation=None):
        delivered = False
        try:
            self.draw(['Quezon City', 'Checking weather...', '', '', '', 'Open-Meteo.com'], generation)
            try:
                if self.cached is None or time.monotonic() - self.cached_at >= 300:
                    self.cached = self.fetch()
                    self.cached_at = time.monotonic()
                lines = ['Quezon City', *self.cached, '', 'Open-Meteo.com']
                temperature = self.cached[0].split(' C', 1)[0]
                condition = f'Quezon City, {temperature} degrees Celsius, {self.cached[1]}.'
            except Exception as exc:
                condition = None
                print(f'Weather unavailable: {exc}', flush=True)
                lines = ['Quezon City', 'Weather unavailable', 'Check internet', 'Try again later', '', 'Open-Meteo.com']
            self.draw(lines, generation)
            executed_at = time.monotonic()
            if condition and self.on_weather_ready and not self.closed and generation == self.generation:
                # Fetch and synthesis stay off the microphone processing thread.
                try:
                    audio, rate = synthesize_response_audio(condition)
                    if not self.closed and generation == self.generation:
                        self.on_weather_ready((generation, condition, audio, rate, executed_at))
                        delivered = True
                except Exception as exc:
                    print(f'Weather speech unavailable: {exc}', flush=True)
        except Exception as exc:
            print(f'OLED unavailable: {exc}', flush=True)
        finally:
            if not delivered and self.on_weather_ready and not self.closed and generation == self.generation:
                self.on_weather_ready((generation, None, None, None, None))

    def close(self):
        with self.lock:
            self.closed = True
            self.timer_cancel.set()
            self.display_event.set()
        self.display_worker.join(timeout=1.)



# Credentials live outside exported model folders and are loaded on each command.
TELEGRAM_MESSAGES = (
    'I spoke. Kibo understood. This message happened. \U0001f916 Please clap.',
    'It works! \U0001f389 Those hours of training finally produced something besides a hot GPU.',
    'My model has graduated from confidently guessing. \U0001f393 Today, it understood the assignment!',
    'From sound waves to Telegram. \U0001f916 My model is working, and I am accepting applause.',
    'Kibo here! Have a great day! \U0001f916\u2600\ufe0f',
    'Enjoy your day! Take a moment to do something that makes you smile. \U0001f60a',
    'A little hello from Kibo. Wishing you a wonderful day! \U0001f916',
    'Hope your day is going well! Remember to take a break and drink some water. \U0001f4a7',
    'Good vibes delivered by voice command. Have an awesome day! \u2728',
    'You have got this! Wishing you a productive and happy day. \U0001f31f',
    'Take it one step at a time, and enjoy the little things today. \U0001f60a',
    'Kibo says hello! May your day have fewer errors and more smiles. \U0001f916',
    'Sending a little sunshine your way. Enjoy the rest of your day! \u2600\ufe0f',
    'Voice command received, good wishes delivered. Have a lovely day! \U0001f916',
)

def send_telegram_message():
    import json
    import urllib.request
    import urllib.error
    config_path = Path.home() / '.config' / 'kibo' / 'telegram.json'
    try:
        config = json.loads(config_path.read_text(encoding='utf-8'))
        token = config['bot_token']
        chat_id = config['chat_id']
        if not isinstance(token, str) or not token.strip() or not chat_id:
            raise ValueError('Missing credentials')
    except (OSError, ValueError, KeyError, TypeError):
        raise RuntimeError('Telegram is not configured. Run python3 ~/setup-kibo-telegram.py first.') from None
    import random
    payload = json.dumps({'chat_id': chat_id, 'text': random.choice(TELEGRAM_MESSAGES)}).encode('utf-8')
    request = urllib.request.Request(
        'https://api.telegram.org/bot' + token + '/sendMessage',
        data=payload, headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError('Telegram rejected the message (HTTP %s). Check bot setup.' % exc.code) from None
    except Exception:
        # Never print request URLs: Telegram embeds the credential in the URL.
        # Do not retry automatically: a timed-out request may already be delivered.
        raise RuntimeError('Telegram delivery could not be confirmed. No automatic retry was made.') from None
    if not isinstance(result, dict) or result.get('ok') is not True:
        raise RuntimeError('Telegram did not confirm the message.')
    return 'Telegram message sent.'

def main():
    print(flush=True)
    default_model = default_paths(Path(__file__).resolve().parent)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, default=default_model)
    ap.add_argument('--responses-dir', type=Path, default=responses_path(Path(__file__).resolve().parent))
    ap.add_argument('--music-dir', type=Path, default=Path(__file__).resolve().parent / 'music')
    ap.add_argument('--music-volume', type=int, default=50, help='Initial music volume, 0-100')
    ap.add_argument('--timeout', type=float, default=30., help='Seconds to wait for one command after either wake phrase while music plays')
    ap.add_argument('--speech-threshold', type=float, default=.015)
    ap.add_argument('--silence-duration', type=float, default=.3, help='End-of-speech silence wait in seconds; 0 ends on the first quiet audio block')
    ap.add_argument('--max-recording', type=float, default=10.)
    ap.add_argument('--pre-speech', type=float, default=.2)
    ap.add_argument('--post-speech', type=float, default=.2)
    args = ap.parse_args()
    if not 0 <= args.music_volume <= 100:
        ap.error('Music volume must be between 0 and 100.')
    if min(args.timeout, args.speech_threshold, args.max_recording) <= 0 or min(args.silence_duration, args.pre_speech, args.post_speech) < 0:
        ap.error('Timeout and recording limit and speech threshold must be positive; silence duration and audio margins cannot be negative.')
    if args.max_recording <= args.silence_duration:
        ap.error('--max-recording must exceed --silence-duration.')
    command_model, checkpoint = load_model(args.model)
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit('Install python-sounddevice in the environment on your microphone computer.')
    responses = {label: read_response(args.responses_dir / filename)
                 for label, filename in RESPONSE_FILES.items()
                 if (label == 'WAKE' or (label not in MUSIC_COMMANDS and
                     label in {name.upper() for name in checkpoint['labels']}))
                 and response_available(label, args.responses_dir / filename)}
    if 'MESSAGE' in responses:
        responses['MESSAGE_SENT'] = read_response(args.responses_dir / 'MESSAGE/message_sent_check_telegram.wav')
    player = MusicPlayer(args.music_dir, args.music_volume)
    weather = WeatherDisplay()
    timer_events = queue.Queue()
    weather.on_timer_done = timer_events.put
    weather_events = queue.Queue()
    weather.on_weather_ready = weather_events.put
    torch.set_num_threads(4)

    chunks = queue.Queue(maxsize=1000)
    responding = threading.Event()
    microphone_gate = MicrophoneGate()
    def callback(indata, frames, time_info, status):
        if responding.is_set():
            return
        if status:
            print(f'Microphone: {status}', flush=True)
        try:
            adc_time = float(time_info.inputBufferAdcTime)
            lag = max(0., float(time_info.currentTime) - adc_time) if adc_time > 0 else 0.
            chunks.put_nowait((time.monotonic() - lag, indata[:, 0].copy()))
        except queue.Full:
            pass

    block_samples = max(1, int(SR * .02))
    pre = deque(maxlen=max(1, int(args.pre_speech * SR / block_samples)))
    session = MusicCommandGate(player, args.timeout)
    music_reminder_shown = False
    startup_prompt_shown = False
    def listening_prompt():
        nonlocal music_reminder_shown, startup_prompt_shown
        session.refresh(time.monotonic())
        waiting_for_wake = session.activated and player.state == 'playing' and session.deadline is None
        if waiting_for_wake:
            if not music_reminder_shown:
                print(f'\nMusic is playing. Say {WAKE_PROMPT} before the next command.', flush=True)
            music_reminder_shown = True
            return
        music_reminder_shown = False
        if not session.activated:
            if not startup_prompt_shown:
                print(f'Say {WAKE_PROMPT} to activate.', flush=True)
                startup_prompt_shown = True
        elif player.state != 'playing':
            print('\nListening for commands...', flush=True)
        elif session.deadline is not None:
            print('Music softened to at most 5%. Listening for one command...\n', flush=True)

    def clear_audio_queue():
        while True:
            try:
                chunks.get_nowait()
            except queue.Empty:
                break
        pre.clear()

    response_lines = []

    response_line_open = False

    def show_response_text(response=None):
        nonlocal response_line_open
        text = response if response is not None else ' '.join(response_lines)
        if response is None:
            response_lines.clear()
        text = ' '.join(text.split())
        if text:
            print((' ' if response_line_open else '[RESPONSE] ') + text, end='', flush=True)
            response_line_open = True

    def report_response_latency(started, ended=None, endpoint='response complete', response=None):
        nonlocal response_line_open
        if started is not None:
            elapsed = max(0., (time.monotonic() if ended is None else ended) - started)
            show_response_text(response)
            prefix = ' | ' if response_line_open else '[RESPONSE] '
            print(f'{prefix}End-to-end latency: {elapsed:.3f} s', flush=True)
            response_line_open = False

    weather_command_starts = {}
    command_executed_at = None

    def play_response(label, command=False):
        nonlocal response_line_open, command_executed_at
        command_executed_at = None
        label = label.upper()

        def say_answer(audio, rate):
            if command and label != 'WEATHER':
                show_response_text()
            player.say(audio, rate)
        responding.set()
        try:
            with player.interruption(hold=label in MUSIC_COMMANDS):
                if command:
                    response_lines.clear()
                    screen_response = weather.handle(label)
                    if screen_response:
                        response_lines.append(screen_response)
                    if label == 'MESSAGE':
                        response_lines.append('Sending message.')
                        say_answer(*responses['MESSAGE'])
                        message = send_telegram_message()
                        command_executed_at = time.monotonic()
                        response_lines.append(message)
                        message = None  # Collected for the final response line.
                        say_answer(*responses['MESSAGE_SENT'])
                    else:
                        message = lights.handle(label) if label in LED_COMMANDS else player.handle(label)
                        command_executed_at = time.monotonic()
                    weather.refresh_music(player, force=True)
                    if message:
                        response_lines.append(message)
                        if label in MUSIC_COMMANDS:
                            show_response_text()
                if command and label == 'CALL':
                    response_lines.append('Making a call.')
                if label == 'TIME':
                    text = spoken_time_text()
                    command_executed_at = time.monotonic()
                    response_lines.append(text)
                    say_answer(*synthesize_response_audio(text))
                elif command and label == 'LIST_REMINDERS':
                    for filename in reminder_response_files(tuple(weather.reminders)):
                        say_answer(*read_response(args.responses_dir / filename))
                elif label not in MUSIC_COMMANDS and label in responses and not (command and label == 'MESSAGE'):
                    audio, rate = responses[label]
                    say_answer(audio, rate)
                if command and label == 'CALL':
                    response_lines.append("Nobody's home right now.")
                    say_answer(*read_response(args.responses_dir / 'CALL' / 'ringing.wav'))
                    say_answer(*read_response(args.responses_dir / 'CALL' / 'nobodys_home_right_now.wav'))
        except Exception as exc:
            if response_line_open:
                print(flush=True)
                response_line_open = False
            print(f'Could not complete {label}: {exc}', flush=True)
            return False
        finally:
            if command:
                session.complete()
            # Only PAUSE and STOP wait for the speaker tail and a quiet boundary.
            if command and label in {'PAUSE', 'STOP'}:
                latency = float(player._stream.latency) if player._stream is not None else 0.
                microphone_gate.reset(settle=latency + .25)
            clear_audio_queue()
            responding.clear()

        return True

    def play_weather_condition(event):
        generation, condition, audio, rate, executed_at = event
        started = weather_command_starts.pop(generation, None)
        if weather.closed or generation != weather.generation:
            return
        if condition is None:
            listening_prompt()
            return
        responding.set()
        try:
            with player.interruption(hold=False):
                show_response_text(condition)
                player.say(audio, rate)
            report_response_latency(started, executed_at)
        except Exception as exc:
            print(f'Could not speak weather: {exc}', flush=True)
        finally:
            clear_audio_queue()
            responding.clear()
            listening_prompt()


    def play_timer_alarm():
        responding.set()
        try:
            with player.interruption(hold=False):
                player.say(*timer_alarm_audio(player.rate))
        except Exception as exc:
            print(f'Could not play timer alarm: {exc}', flush=True)
        finally:
            # Ignore the alarm itself and any delayed speaker audio.
            latency = float(player._stream.latency) if player._stream is not None else 0.
            microphone_gate.reset(settle=latency + .25)
            clear_audio_queue()
            responding.clear()

    speech, silent, total = [], 0, 0
    onset_samples = 0
    onset_started = None
    command_started = None
    onset_needed = int(.12 * SR)
    active_at_start = False
    quiet_at_start = False
    silence_needed = max(1, int(args.silence_duration * SR))  # Zero wait still requires a quiet block.
    max_samples = int(args.max_recording * SR)
    post_needed = int(args.post_speech * SR)
    lights = None
    try:
        lights = LEDController()
        player.start(sd)
        with sd.InputStream(samplerate=SR, channels=1, dtype='float32', blocksize=block_samples, callback=callback):
            listening_prompt()
            while True:
                weather.refresh_music(player)
                # Use the main thread so alarm and voice responses never overlap.
                try:
                    finished_timer = timer_events.get_nowait()
                except queue.Empty:
                    finished_timer = None
                if finished_timer is not None and not finished_timer.is_set():
                    speech, silent, total = [], 0, 0
                    onset_samples = 0
                    play_timer_alarm()
                    listening_prompt()
                    continue
                # Serialize speech with other responses; discard stale weather.
                if not speech:
                    try:
                        weather_event = weather_events.get_nowait()
                    except queue.Empty:
                        weather_event = None
                    if weather_event is not None:
                        play_weather_condition(weather_event)
                try:
                    captured, block = chunks.get(timeout=.1)
                except queue.Empty:
                    if not speech and session.refresh(time.monotonic()):
                        listening_prompt()
                    continue
                rms = float(np.sqrt(np.mean(block * block) + 1e-12))
                if not microphone_gate.ready(captured, rms, len(block), args.speech_threshold):
                    speech, silent, total = [], 0, 0
                    onset_samples = 0
                    pre.clear()
                    continue
                if not speech:
                    if session.refresh(captured):
                        listening_prompt()
                    rms = float(np.sqrt(np.mean(block * block) + 1e-12))
                    if rms < args.speech_threshold:
                        onset_samples = 0
                        pre.append(block)
                        continue
                    if onset_samples == 0:
                        # ADC timestamp of the first above-threshold microphone block.
                        onset_started = captured
                    onset_samples += len(block)
                    if onset_samples < onset_needed:
                        pre.append(block)
                        continue
                    onset_samples = 0
                    command_started = onset_started
                    active_at_start = session.accepts(captured)
                    quiet_at_start = not active_at_start
                    if not quiet_at_start:
                        print('Speech detected.', flush=True)
                    speech = list(pre)
                    pre.clear()
                    silent = total = 0
                speech.append(block)
                total += len(block)
                rms = float(np.sqrt(np.mean(block * block) + 1e-12))
                silent = silent + len(block) if rms < args.speech_threshold else 0
                if silent < silence_needed and total < max_samples:
                    continue
                audio = np.concatenate(speech)
                speech = []
                if silent < silence_needed:
                    if not quiet_at_start:
                        print('Recording limit reached; utterance ignored.', flush=True)
                    listening_prompt()
                    continue
                if silent > post_needed:
                    audio = audio[:-(silent - post_needed)]
                x = wav_to_logmel(torch.from_numpy(audio)).unsqueeze(0)
                with torch.no_grad():
                    inference_started = time.perf_counter()
                    logits = command_model(x)
                    inference_ms = (time.perf_counter() - inference_started) * 1000
                    probs = logits.softmax(1)[0]
                    confidence, index = probs.max(0)
                    label = checkpoint['labels'][index.item()]
                    # Ignore legacy wake predictions defensively.
                    if label == 'KNOCK_KNOCK':
                        continue
                    if not session.activated and label not in WAKE_PHRASES:
                        continue
                    if label == EXIT_LABEL:
                        weather.dismiss()
                        player.handle('STOP')
                        print('\nSAGITTARIUS detected. Stopping the program.\n', flush=True)
                        break
                    elif label in WAKE_PHRASES:
                        initial_wake = not session.activated
                        session.wake(time.monotonic())
                        if initial_wake:
                            play_response('WAKE')
                        else:
                            clear_audio_queue()
                        print(f'\n{WAKE_PHRASES[label].upper()} detected.', flush=True)
                    elif active_at_start:
                        print(f'[INPUT] {label} | Confidence: {confidence.item():.1%} | Inference Latency: {inference_ms:.2f} ms', flush=True)
                        completed = play_response(label, command=True)
                        if label == 'WEATHER':
                            weather_command_starts.clear()
                            if completed:
                                weather_command_starts[weather.generation] = command_started
                        elif completed:
                            if label.upper() in {'TIMER_10S', 'TIMER_30S', 'TIMER_1M'}:
                                if weather.timer_started_at is not None:
                                    report_response_latency(command_started, weather.timer_started_at, 'countdown start')
                            else:
                                report_response_latency(command_started, command_executed_at)
                if not (active_at_start and label == 'WEATHER'):
                    listening_prompt()
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        try:
            session.complete()
            player.close()
        finally:
            weather.close()
            if lights is not None:
                lights.close()


if __name__ == '__main__':
    with single_instance():
        main()
