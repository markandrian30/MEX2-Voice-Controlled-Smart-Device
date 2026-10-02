import torch
import torch.nn as nn

MODEL_VERSION = "voicecnn_length_aware_v2"


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
