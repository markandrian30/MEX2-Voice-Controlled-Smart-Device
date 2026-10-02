"""Length-aware CNN + bidirectional GRU for complete spoken commands."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from model import LengthAwareBlock

MODEL_VERSION = "voicecrnn_length_aware_v1"


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
