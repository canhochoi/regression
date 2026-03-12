import torch
import torch.nn as nn
from typing import Sequence, Optional, Callable, Union, List

ActivationFactory = Callable[[], nn.Module]

def make_mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    activation: ActivationFactory = nn.ReLU,
    dropout: float = 0.0,
    batch_norm: bool = False,
    layer_norm: bool = False,
    bias: bool = True,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h, bias=bias))
        if batch_norm:
            layers.append(nn.BatchNorm1d(h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation())
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim, bias=bias))
    return nn.Sequential(*layers)

class MLPEncoderDecoder(nn.Module):
    """
    Encoder: x -> z (latent)
    Decoder: z -> y_hat
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        latent_dim: int = 64,
        enc_hidden: Sequence[int] = (512, 256),
        dec_hidden: Sequence[int] = (256, 512),
        activation: ActivationFactory = nn.ReLU,
        dropout: float = 0.0,
        batch_norm: bool = False,
        layer_norm: bool = False,
        bias: bool = True,
        out_activation: Optional[nn.Module] = None,  # e.g., Softmax(dim=1) if predicting proportions
    ):
        super().__init__()
        self.encoder = make_mlp(
            in_dim=in_features,
            hidden=enc_hidden,
            out_dim=latent_dim,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            bias=bias,
        )
        self.decoder = make_mlp(
            in_dim=latent_dim,
            hidden=dec_hidden,
            out_dim=out_features,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            bias=bias,
        )
        self.out_activation = out_activation

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        y = self.decoder(z)
        if self.out_activation is not None:
            y = self.out_activation(y)
        return y, z  # return latent too (useful for diagnostics)