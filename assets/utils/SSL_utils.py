import torch
import torch.nn as nn

def mask_landmarks(x, mask_ratio=0.15):
    """
    x: [B, T, V, C]
    C could be x, y, z, confidence, velocity, etc.
    """
    mask = torch.rand(x.shape[:-1], device=x.device) < mask_ratio
    x_masked = x.clone()
    x_masked[mask] = 0.0

    return x_masked, mask



# A basic pretraining model:

class MaskedLandmarkModel(nn.Module):
    def __init__(self, encoder, hidden_dim, input_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x_masked):
        """
        x_masked: [B, T, V, C]
        """
        #features = self.encoder(x_masked)  # [B, T, V, D]
        features = self.encoder.encode(x_masked)
        reconstructed = self.decoder(features)  # [B, T, V, C]
        return reconstructed



# After pretraining, reuse the encoder weights for classification:









