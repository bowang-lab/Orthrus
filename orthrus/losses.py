from typing import Callable, Optional, Tuple

import numpy as np

from torchmetrics import Accuracy

import torch
from torch import Tensor
from torch import distributed as torch_dist
from torch import nn


class DCLLoss(nn.Module):
    """Implementation of the Decoupled Contrastive Learning Loss from
    Decoupled Contrastive Learning [0].

    This code implements Equation 6 in [0], including the sum over all images
    `i` and views `k`. The loss is reduced to a mean loss over the mini-batch.
    The implementation was inspired by [1].

    - [0] Chun-Hsiao Y. et. al., 2021, Decoupled Contrastive Learning
          https://arxiv.org/abs/2110.06848
    - [1] https://github.com/raminnakhli/Decoupled-Contrastive-Learning

    Attributes:
        temperature:
            Similarities are scaled by inverse temperature.
        weight_fn:
            Weighting function `w` from the paper. Scales the loss between the
            positive views (views from the same image). No weighting is
            performed if weight_fn is None. The function must take the two
            input tensors passed to the forward call as input and return a
            weight tensor. The returned weight tensor must have the same length
            as the input tensors.
        gather_distributed:
            If True then negatives from all gpus are gathered before the
            loss calculation.

    Examples:

        >>> loss_fn = DCLLoss(temperature=0.07)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # embed images using some model, for example SimCLR
        >>> out0 = model(t0)
        >>> out1 = model(t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
        >>>
        >>> # you can also add a custom weighting function
        >>> weight_fn = lambda out0, out1: torch.sum((out0 - out1) ** 2, dim=1)
        >>> loss_fn = DCLLoss(weight_fn=weight_fn)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        weight_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        gather_distributed: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn
        self.gather_distributed = gather_distributed

        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not"
                " available. Please set gather_distributed=False or install a "
                "torch version with distributed support."
            )

    def forward(
        self,
        out0: torch.Tensor,
        out1: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward pass of the DCL loss.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)

        Returns:
            Mean loss over the mini-batch and batch metrics.
        """
        # normalize the output to length 1
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        if self.gather_distributed and world_size() > 1:
            # gather representations from other processes if necessary
            out0_all = torch.cat(gather(out0), 0)
            out1_all = torch.cat(gather(out1), 0)
        else:
            out0_all = out0
            out1_all = out1

        # calculate symmetric loss
        loss0, metrics0 = self._loss(out0, out1, out0_all, out1_all)
        loss1, metrics1 = self._loss(out1, out0, out1_all, out0_all)
        loss = 0.5 * (loss0 + loss1)

        metrics = {
            "mean_loss": loss.item(),
            "positive_pair_similarity": np.mean([
                metrics0["positive_pair_similarity"],
                metrics1["positive_pair_similarity"]
            ]),
            "negative_pair_similarity": np.mean([
                metrics0["negative_pair_similarity"],
                metrics1["negative_pair_similarity"]
            ]),
            "contrastive_accuracy": np.mean([
                metrics0["contrastive_accuracy"],
                metrics1["contrastive_accuracy"]
            ]),
        }
        return loss, metrics

    def _loss(
        self,
        out0: Tensor,
        out1: Tensor,
        out0_all: Tensor,
        out1_all: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        """Calculates DCL loss for out0 with respect to its positives in out1
        and the negatives in out1, out0_all, and out1_all.

        This code implements Equation 6 in [0], including the sum over all
        images `i` but with `k` fixed at 0.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)
            out0_all:
                Output projections of the first set of transformed images from
                all distributed processes/gpus. Should be equal to out0 in an
                undistributed setting.
                Shape (batch_size * world_size, embedding_size)
            out1_all:
                Output projections of the second set of transformed images from
                all distributed processes/gpus. Should be equal to out1 in an
                undistributed setting.
                Shape (batch_size * world_size, embedding_size)

        Returns:
            Mean loss over the mini-batch and metrics.
        """
        # create diagonal mask that only selects similarities between
        # representations of the same images
        batch_size = out0.shape[0]
        global_bs = gather_batch_sizes(out0)

        acc = Accuracy(
            task="multiclass",
            num_classes=sum(global_bs)
        ).to(out0.device)

        offset = sum(global_bs[: rank()])
        labels_idx = (torch.arange(batch_size) + offset).to(out0.device)

        diag_mask = eye_rank_irregular(out0)

        # calculate similarities
        # here n = batch_size and m = batch_size * world_size.
        sim_00 = torch.einsum("nc,mc->nm", out0, out0_all) / self.temperature
        sim_01 = torch.einsum("nc,mc->nm", out0, out1_all) / self.temperature

        positive_loss = -sim_01[diag_mask]
        if self.weight_fn:
            positive_loss = positive_loss * self.weight_fn(out0, out1)

        accuracy = acc(sim_01, labels_idx).cpu().item()
        # Mean similarity of positive pairs
        positive_similarity = sim_01.diag().mean().item()
        # Approx. mean negative similarity
        negative_similarity = (sim_00.mean() + sim_01[~diag_mask].mean()) / 2.0

        metrics = {
            "contrastive_accuracy": accuracy,
            "positive_pair_similarity": positive_similarity,
            "negative_pair_similarity": negative_similarity,
        }

        # remove simliarities between same views of the same image
        sim_00 = sim_00[~diag_mask].view(batch_size, -1)
        # remove similarities between different views of the same images
        # this is the key difference compared to NTXentLoss
        sim_01 = sim_01[~diag_mask].view(batch_size, -1)

        negative_loss_00 = torch.logsumexp(sim_00, dim=1)
        negative_loss_01 = torch.logsumexp(sim_01, dim=1)

        total_loss = positive_loss + negative_loss_00 + negative_loss_01
        total_loss = total_loss.mean()

        return total_loss, metrics


def rank() -> int:
    """Returns the rank of the current process."""
    return torch_dist.get_rank() if torch_dist.is_initialized() else 0


def world_size() -> int:
    """Returns the current world size (number of distributed processes)."""
    return torch_dist.get_world_size() if torch_dist.is_initialized() else 1


def gather(input: torch.Tensor) -> Tuple[torch.Tensor]:
    """Gathers this tensor from all processes. Supports backprop."""
    return GatherIrregularLayer.apply(input)


class GatherIrregularLayer(torch.autograd.Function):
    """Gather irregular tensors from all processes, supporting backprop.

    Supports the case where gathered tensors are not regularly shaped.
    For example, given a rank 0 tensor of shape (B_1 x D) and a rank 1 tensor
    of shape (B_2 x D), returns a tensor of shape ((B_1 + B_2) x D).
    """

    @staticmethod
    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        self.save_for_backward(input)

        world_size = torch_dist.get_world_size()

        all_bs = gather_batch_sizes(input)
        size_diff = max(all_bs) - input.size(0)

        if size_diff:
            padding = torch.zeros(
                (size_diff, input.size(1)),
                device=input.device,
                dtype=input.dtype
            )

            input = torch.cat((input, padding), dim=0)

        all_in_pad = [torch.zeros_like(input) for _ in range(world_size)]
        torch_dist.all_gather(all_in_pad, input)

        all_input = []
        for input, bs in zip(all_in_pad, all_bs):
            all_input.append(input[:bs])
        return all_input

    @staticmethod
    def backward(self, *grads: torch.Tensor) -> torch.Tensor:
        (input,) = self.saved_tensors
        grad_out = torch.empty_like(input)
        grad_out[:] = grads[torch_dist.get_rank()]
        return grad_out


def eye_rank_irregular(ex_in: torch.Tensor) -> torch.Tensor:
    """Compute the zero matrix with diagonal for process rank set to one.

    Supports exmaples where each process may have different batch sizes.
    """
    local_bs = ex_in.shape[0]
    global_bs = gather_batch_sizes(ex_in)

    rows = torch.arange(local_bs, device=ex_in.device, dtype=torch.long)
    cols = rows + sum(global_bs[: rank()])

    diag_mask = torch.zeros((local_bs, sum(global_bs)), dtype=torch.bool)
    diag_mask[(rows, cols)] = True

    return diag_mask


def gather_batch_sizes(tensor: torch.Tensor) -> list[int]:
    """Gather batch sizes for a tensor across all processes.

    Used when batches have different sizes across GPUs.

    Args:
        tensor: Local input tensor.
    Returns:
        List of first dimension shape of tensors across processes.
    """
    if not torch_dist.is_initialized():
        return [tensor.size(0)]

    world_size = torch_dist.get_world_size()

    local_bs = torch.tensor(tensor.size(0), device=tensor.device)
    all_bs = [torch.zeros_like(local_bs) for _ in range(world_size)]

    torch_dist.all_gather(all_bs, local_bs)

    all_bs = [bs.item() for bs in all_bs]
    return all_bs
