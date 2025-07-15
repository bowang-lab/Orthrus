from functools import partial
from typing import Callable, Optional, Tuple
from torchmetrics import Accuracy, Metric

import torch
from torch import Tensor
from torch import distributed as torch_dist
from torch import nn
from torch.nn import functional as F


def negative_mises_fisher_weights(
    out0: Tensor, out1: Tensor, sigma: float = 0.5
) -> torch.Tensor:
    """Negative Mises-Fisher weighting function as presented in Decoupled
    Contrastive Learning [0].

    The implementation was inspired by [1].

    - [0] Chun-Hsiao Y. et. al., 2021, Decoupled Contrastive Learning https://arxiv.org/abs/2110.06848
    - [1] https://github.com/raminnakhli/Decoupled-Contrastive-Learning

    Args:
        out0:
            Output projections of the first set of transformed images.
            Shape: (batch_size, embedding_size)
        out1:
            Output projections of the second set of transformed images.
            Shape: (batch_size, embedding_size)
        sigma:
            Similarities are scaled by inverse sigma.
    Returns:
        A tensor with shape (batch_size,) where each entry is the weight for one
        of the input images.

    """
    similarity = torch.einsum("nm,nm->n", out0.detach(), out1.detach()) / sigma
    return 2 - out0.shape[0] * nn.functional.softmax(similarity, dim=0)

class PairingTypeCounter(Metric):
    """A metric to count the occurrences of different pairing types."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # From collate_fn: {"self": 0, "splice": 1, "ortho": 2}
        self.add_state("counts", default=torch.zeros(3), dist_reduce_fx="sum")

    def update(self, pairing_types: Tensor) -> None:
        """Update state with pairing types from a batch."""
        # Get the counts of each type in the current batch
        types, counts = torch.unique(pairing_types, return_counts=True)
        # Add the counts to the correct index in our state
        for type_val, count_val in zip(types, counts):
            self.counts[type_val] += count_val

    def compute(self) -> dict[str, Tensor]:
        """Compute the final counts."""
        return {
            "self_pairs": self.counts[0],
            "splice_pairs": self.counts[1],
            "ortho_pairs": self.counts[2],
        }

class DINOLoss(nn.Module):
    """Implementation of the DINO loss from Emerging Properties in Self-Supervised
    Vision Transformers [0].

    This code implements Equation 3 in [0], following the implementation in [1].

    - [0] Caron M. et. al., 2021, Emerging Properties in Self-Supervised Vision Transformers https://arxiv.org/abs/2104.14294
    - [1] https://github.com/facebookresearch/dino
    """
    def __init__(
        self,
        representation_dim: int = 1024,
        student_temperature: float = 0.1,
        warmup_teacher_temperature: float = 0.04,
        warmup_teacher_temperature_epochs: int = 30,
        teacher_temperature: float = 0.04,
        center_momentum: float = 0.9,
        num_local_views: int = 8,
        num_global_views: int = 2,
        total_epochs: int = 300,
    ):
        """
        TODO: Write description
        """
        super().__init__()
        self.student_temperature = student_temperature
        self.center_momentum = center_momentum
        self.num_augs = (num_local_views * num_global_views) + num_global_views
        self.num_global_views = num_global_views

        # initialize the center as a learnable parameter
        # updated as an exponential moving average 
        self.register_buffer(
            "center",
            torch.zeros((1, representation_dim)),
        )

        # teacher temperture has a warmup phase avoiding initial instability
        # and then remains constant afterwards
        self.teacher_temperature_schedule = torch.cat(
            (
                torch.linspace(
                    start = warmup_teacher_temperature,
                    end = teacher_temperature,
                    steps = warmup_teacher_temperature_epochs
                ),
                torch.tensor([
                    teacher_temperature
                ]).repeat(total_epochs - warmup_teacher_temperature_epochs)
            )
        )

    def forward(
        self,
        student_output: torch.Tensor, # (num_images * num_augs, representation_dim)
        teacher_output: torch.Tensor, # (num_images * num_global_views, representation_dim)
        epoch: int,
    ) -> torch.Tensor:
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        # teacher centering + sharpening
        teacher_softmax = F.softmax(
            (teacher_output - self.center) / self.teacher_temperature_schedule[epoch], 
            dim=-1
        ).detach().view(self.num_global_views, -1, teacher_output.shape[-1])

        # FIXME: this might be less efficient than the original implementation
        #        because we store the student softmax in memory
        # student sharpening
        student_softmax = F.log_softmax(
            student_output / self.student_temperature,
            dim=-1
        ).view(self.num_augs, -1, student_output.shape[-1])

        # g -> num global views, b -> batch size, d -> representation dimension
        # a -> num local augmentations + global views
        loss_matrix = torch.einsum(
            "gbd,abd -> gab",
            -teacher_softmax, student_softmax
        )
        mask = torch.ones((self.num_global_views, self.num_augs), dtype=torch.bool, device=loss_matrix.device)
        
        # mask out the diagonal elements corresponding to the same view
        for i in range(self.num_global_views):
            mask[i, i] = False
        loss = loss_matrix[mask].mean()

        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(
        self,
        teacher_output: torch.Tensor,
    ):
        """
        Update the center of the momentum encoder using the teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * world_size())

        # ema update
        # momentum*center + (1-momentum)*new_center
        self.center = self.center_momentum * self.center + (1 - self.center_momentum) * batch_center

class DCLLoss(nn.Module):
    """Implementation of the Decoupled Contrastive Learning Loss from
    Decoupled Contrastive Learning [0].

    This code implements Equation 6 in [0], including the sum over all images `i`
    and views `k`. The loss is reduced to a mean loss over the mini-batch.
    The implementation was inspired by [1].

    - [0] Chun-Hsiao Y. et. al., 2021, Decoupled Contrastive Learning https://arxiv.org/abs/2110.06848
    - [1] https://github.com/raminnakhli/Decoupled-Contrastive-Learning

    Attributes:
        temperature:
            Similarities are scaled by inverse temperature.
        weight_fn:
            Weighting function `w` from the paper. Scales the loss between the
            positive views (views from the same image). No weighting is performed
            if weight_fn is None. The function must take the two input tensors
            passed to the forward call as input and return a weight tensor. The
            returned weight tensor must have the same length as the input tensors.
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
        self_weight: float = 1.0,
        splice_weight: float = 1.0,
        ortho_weight: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn
        self.gather_distributed = gather_distributed
        self.self_weight = self_weight
        self.splice_weight = splice_weight
        self.ortho_weight = ortho_weight
        # From collate_fn: {"self": 0, "splice": 1, "ortho": 2}
        self.pairing_weights = torch.tensor([self.self_weight, self.splice_weight, self.ortho_weight])

        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(
        self,
        out0: Tensor,
        out1: Tensor,
        pairing_types: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of the DCL loss.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)
            pairing_types:
                Tensor with integer types for each pair.
                Shape: (batch_size,)

        Returns:
            Mean loss over the mini-batch.
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
        loss0, metrics0 = self._loss(out0, out1, out0_all, out1_all, pairing_types)
        loss1, metrics1 = self._loss(out1, out0, out1_all, out0_all, pairing_types)
        loss = 0.5 * (loss0 + loss1)
        metrics = {
            'mean_loss': loss.item(),
            'positive_pair_similarity': (metrics0['positive_pair_similarity'] + metrics1['positive_pair_similarity']) / 2,
            'negative_pair_similarity': (metrics0['negative_pair_similarity'] + metrics1['negative_pair_similarity']) / 2,
            'accuracy': (metrics0['accuracy'] + metrics1['accuracy']) / 2,
        }
        return loss, metrics

    def _loss(self, out0, out1, out0_all, out1_all, pairing_types: Optional[Tensor] = None):
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
            pairing_types:
                Tensor with integer types for each pair.
                Shape: (batch_size,)

        Returns:
            Mean loss over the mini-batch.
        """
        # create diagonal mask that only selects similarities between
        # representations of the same images
        batch_size = out0.shape[0]
        global_bs = gather_batch_sizes(out0)

        acc = Accuracy(
            task='multiclass',
            num_classes=sum(global_bs)
        ).to(out0.device)

        offset = sum(global_bs[:rank()])
        labels_idx = (torch.arange(batch_size) + offset).to(out0.device)

        diag_mask = eye_rank_irregular(out0)

        # calculate similarities
        # here n = batch_size and m = batch_size * world_size.
        sim_00 = torch.einsum("nc,mc->nm", out0, out0_all) / self.temperature
        sim_01 = torch.einsum("nc,mc->nm", out0, out1_all) / self.temperature

        positive_loss = -sim_01[diag_mask]
        if self.weight_fn:
            positive_loss = positive_loss * self.weight_fn(out0, out1)

        if pairing_types is not None:
            self.pairing_weights = self.pairing_weights.to(positive_loss.device)
            type_weights = self.pairing_weights[pairing_types]
            positive_loss = positive_loss * type_weights

        accuracy = acc(sim_01, labels_idx).cpu()
        # Mean similarity of positive pairs
        positive_similarity = sim_01.diag().mean().item()
        # Approx. mean negative similarity
        negative_similarity = (sim_00.mean() + sim_01[~diag_mask].mean()) / 2.0

        # remove simliarities between same views of the same image
        sim_00 = sim_00[~diag_mask].view(batch_size, -1)
        # remove similarities between different views of the same images
        # this is the key difference compared to NTXentLoss
        sim_01 = sim_01[~diag_mask].view(batch_size, -1)

        negative_loss_00 = torch.logsumexp(sim_00, dim=1)
        negative_loss_01 = torch.logsumexp(sim_01, dim=1)

        total_loss = positive_loss + negative_loss_00 + negative_loss_01
        total_loss = total_loss.mean()

        metrics = {
            'accuracy': accuracy.item(),
            'positive_pair_similarity': positive_similarity,
            'negative_pair_similarity': negative_similarity,
            'positive_loss': positive_loss.mean().item(),
            'negative_loss': (negative_loss_00 + negative_loss_01).mean().item(),
        }

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

def reduce(tensor: torch.Tensor):
    """Reduces a tensor across all processes."""
    if torch_dist.is_initialized():
        torch_dist.all_reduce(tensor)

def gather_batch_sizes(tensor: torch.Tensor) -> list[int]:
    """Gather batch sizes for a tensor across all processes.

    Used when batches have differential size across GPUs.

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


class GatherIrregularLayer(torch.autograd.Function):
    """Gather irregular tensors from all processes, supporting backprop.

    Supports the case where gathered tensors are not regularly shaped.
    For example, given a rank 0 tensor of shape (B_1 x D) and a rank 1 tensor
    of shape (B_2 x D), returns a tensor of shape ((B_1 + B_2) x D).
    """

    @staticmethod
    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """TODO"""
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
        """TODO"""
        (input,) = self.saved_tensors
        grad_out = torch.empty_like(input)
        grad_out[:] = grads[torch_dist.get_rank()]
        return grad_out


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all processes, supporting backward propagation.

    This code was taken and adapted from here:
    https://github.com/Spijkervet/SimCLR
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ctx.save_for_backward(input)
        output = [torch.empty_like(input) for _ in range(torch_dist.get_world_size())]
        torch_dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        grad_out = torch.empty_like(input)
        grad_out[:] = grads[torch_dist.get_rank()]
        return grad_out


def eye_rank_irregular(ex_in: torch.Tensor) -> torch.Tensor:
    """TODO: Fill out description"""
    local_bs = ex_in.shape[0]
    global_bs = gather_batch_sizes(ex_in)

    rows = torch.arange(local_bs, device=ex_in.device, dtype=torch.long)
    cols = rows + sum(global_bs[:rank()])

    diag_mask = torch.zeros((local_bs, sum(global_bs)), dtype=torch.bool)
    diag_mask[(rows, cols)] = True

    return diag_mask


# https://github.com/lightly-ai/lightly/blob/master/lightly/utils/dist.py
def eye_rank(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Returns an (n, n * world_size) zero matrix with the diagonal for the rank
    of this process set to 1.

    Example output where n=3, the current process has rank 1, and there are
    4 processes in total:

        rank0   rank1   rank2   rank3
        0 0 0 | 1 0 0 | 0 0 0 | 0 0 0
        0 0 0 | 0 1 0 | 0 0 0 | 0 0 0
        0 0 0 | 0 0 1 | 0 0 0 | 0 0 0

    Equivalent to torch.eye for undistributed settings or if world size == 1.

    Args:
        n:
            Size of the square matrix on a single process.
        device:
            Device on which the matrix should be created.

    """
    rows = torch.arange(n, device=device, dtype=torch.long)
    cols = rows + rank() * n
    diag_mask = torch.zeros((n, n * world_size()), dtype=torch.bool)
    diag_mask[(rows, cols)] = True
    return diag_mask