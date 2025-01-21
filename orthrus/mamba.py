import math
from functools import partial

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba_simple import Mamba, Block


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mix_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm, eps=norm_epsilon, **factory_kwargs)
    block = Block(
        d_model,
        mix_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        input_dim: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Linear(input_dim, d_model, **factory_kwargs)

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = nn.LayerNorm(d_model, eps=norm_epsilon, **factory_kwargs)

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward(self, x, inference_params=None, channel_last=False):
        if not channel_last:
            x = x.transpose(1, 2)

        hidden_states = self.embedding(x)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(
            residual.to(dtype=self.norm_f.weight.dtype)
        )

        hidden_states = hidden_states

        return hidden_states

    def representation(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        channel_last: bool = False,
        aggregation: str = "mean",
    ) -> torch.Tensor:
        """Get global representation of input data.

        Args:
            x: Data to embed. Has shape (B x C x L) if not channel_last.
            lengths: Unpadded length of each data input.
            channel_last: Expects input of shape (B x L x C).

        Returns:
            Global representation vector of shape (B x H).
        """
        out = self.forward(x, channel_last=channel_last)
        if aggregation == "mean":
            out_tensor = mean_unpadded(out, lengths)
        elif aggregation == "last":
            out_tensor = last_unpadded(out, lengths)
        elif aggregation == "exponential":
            out_tensor = exponential_length_weighting(out, lengths)
        elif aggregation == "exponential_norm":
            out_tensor = exponential_length_weighting(
                out,
                lengths,
                normalized=True
            )
        else:
            raise ValueError(f"Invalid aggregation method: {aggregation}")

        return out_tensor


def exponential_length_weighting(
    x: torch.Tensor,
    lengths: torch.Tensor,
    normalized: bool = False
) -> torch.Tensor:
    """Apply exponential weighting to tensor across second dimension.
    Last element has the highest weighting.

    Args:
        x (torch.Tensor): Tensor to apply weighting. Has shape (B x L x H).
        lengths (torch.Tensor): Tensor of unpadded lengths. Has shape (B).
        normalized (bool): Normalize output by sum of weights.

    Returns:
        torch.Tensor: Weighted tensor of shape (B x H).
    """
    mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
    masked_tensor = x * mask.unsqueeze(-1)
    weights = torch.arange(x.size(1), device=x.device)[None, :] - (lengths - 1)[:, None]
    weights[weights >= 0] = 0
    weights = torch.exp(weights.float())
    weighted_tensor = masked_tensor * weights.unsqueeze(-1)
    sum_tensor = weighted_tensor.sum(dim=1)

    if normalized:
        sum_tensor /= weights.sum(dim=1).unsqueeze(-1)

    return sum_tensor


def last_unpadded(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Get last unpadded element of tensor across second dimension.

    Args:
        x (torch.Tensor): Tensor to get last unpadded element. Has shape (B x L x H).
        lengths (torch.Tensor): Tensor of unpadded lengths. Has shape (B).

    Returns:
        torch.Tensor: Last unpadded element of shape (B x H).
    """
    mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
    masked_tensor = x * mask.unsqueeze(-1)
    last_tensor = masked_tensor[torch.arange(x.size(0), device=x.device), lengths.long() - 1]

    return last_tensor


def mean_unpadded(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Take mean of tensor across second dimension without padding.

    Args:
        x: Tensor to take unpadded mean. Has shape (B x L x H).
        lengths: Tensor of unpadded lengths. Has shape (B)

    Returns:
        Mean tensor of shape (B x H).
    """
    mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
    masked_tensor = x * mask.unsqueeze(-1)
    sum_tensor = masked_tensor.sum(dim=1)
    mean_tensor = sum_tensor / lengths.unsqueeze(-1).float()

    return mean_tensor


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)
