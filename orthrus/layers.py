import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Optimizer
import math

def return_norm_layer(norm_type, num_features):
    if norm_type == "batchnorm":
        return nn.BatchNorm1d(num_features)
    # Add more normalization types if needed
    else:
        return nn.Identity()

class DilatedConv1DBasic(nn.Module):
    def __init__(
        self,
        in_channels,
        filter_num=21,
        kernel_size=8,
        padding='same',
        dropout_prob=0.1,
        dilation=2,
        pooling_layer='max_pool',
        norm_type="batchnorm",
    ):
        super(DilatedConv1DBasic, self).__init__()
        
        self.norm_type = norm_type
        self.filter_num = filter_num

        if in_channels != self.filter_num:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, self.filter_num, kernel_size=1, padding=padding),
                return_norm_layer(self.norm_type, self.filter_num)
            )
        else:
            self.downsample = nn.Identity()
       
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=filter_num, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn1 = return_norm_layer(norm_type, filter_num)
        
        self.conv2 = nn.Conv1d(in_channels=filter_num, out_channels=filter_num, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn2 = return_norm_layer(norm_type, filter_num)

        if dropout_prob > 0:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = nn.Identity()

        if pooling_layer == 'max_pool':
            self.max_pool = nn.MaxPool1d(kernel_size=2)
        # Note: 'attn_pool' and custom pooling layers need to be defined separately
        else:
            self.max_pool = nn.Identity()

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = F.relu(x + residual)
        x = self.max_pool(x)

        return x

class DilatedBasicBlockLayer(nn.Module):
    def __init__(
        self, 
        in_channels, 
        filter_num=21, 
        kernel_size=8, 
        padding='same', 
        dropout_prob=0.1, 
        dilation=2, 
        pooling_layer='max_pool', 
        blocks=2, 
        norm_type="batchnorm", 
        increase_dilation=True
    ):
        super(DilatedBasicBlockLayer, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(blocks):
            # Adjust dilation rate if needed
            current_dilation = dilation * (2**i) if increase_dilation else dilation
            
            # Assuming DilatedConv1DBasic is already defined as discussed
            self.layers.append(
                DilatedConv1DBasic(
                    in_channels=in_channels if i == 0 else filter_num,  # First layer matches input channels, others match filter_num
                    filter_num=filter_num,
                    kernel_size=kernel_size,
                    padding=padding,
                    dropout_prob=dropout_prob, 
                    dilation=current_dilation,
                    pooling_layer=pooling_layer if i == blocks - 1  else '',  # Apply max pooling only the last block
                    norm_type=norm_type,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_features: int,
        projection_body: int,
        projection_head_size: int,
        norm_type: str,
        n_layers: int = 3, 
        output_bias: bool = False,
        output_sigmoid: bool = False,
    ):
        super(ProjectionHead, self).__init__()
        self.output_sigmoid = output_sigmoid
        self.layers = nn.ModuleList()
        
        if n_layers == 1:
            self.layers.append(nn.Linear(input_features, projection_head_size, bias=output_bias))
        else:
            self.layers.append(nn.Linear(input_features, projection_body))
            self.layers.append(return_norm_layer(norm_type, projection_body))
            
            for _ in range(n_layers - 2):
                self.layers.append(nn.Linear(projection_body, projection_body))
                self.layers.append(return_norm_layer(norm_type, projection_body))
            
            self.layers.append(nn.Linear(projection_body, projection_head_size, bias=output_bias))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(0, len(self.layers) - 1, 2):
            x = F.relu(self.layers[i+1](self.layers[i](x)))
        x = self.layers[-1](x)
        if self.output_sigmoid:
            x = torch.sigmoid(x)
        return x
    

class StochasticShift(nn.Module):
    """Stochastically shift a one-hot encoded DNA sequence in PyTorch."""

    def __init__(self, shift_max=3, symmetric=False, pad_value=0):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.symmetric = symmetric
        self.pad_value = pad_value
        if self.symmetric:
            self.augment_shifts = torch.arange(-self.shift_max, self.shift_max + 1)
        else:
            self.augment_shifts = torch.arange(0, self.shift_max + 1)
            
    def forward(self, seq_1hot):
        if self.training:
            shift_i = torch.randint(0, len(self.augment_shifts), (1,)).item()
            shift = self.augment_shifts[shift_i]
            if shift != 0:
                sseq_1hot = self.shift_sequence(seq_1hot, shift)
            else:
                sseq_1hot = seq_1hot
            return sseq_1hot
        else:
            return seq_1hot

    def shift_sequence(self, seq, shift):
        """Shifts the sequence by the specified amount with padding."""
        if seq.dim() != 3:
            raise ValueError("input sequence should be rank 3")
        
        batch_size, channels, seq_length = seq.size()
        pad_size = abs(shift)

        # Create padding tensor
        pad = (torch.ones(batch_size, channels, pad_size) * self.pad_value).to(seq.device)

        if shift > 0:  # shift right
            sliced_seq = seq[:, :, :-shift]
            return torch.cat([pad, sliced_seq], dim=2)
        else:  # shift left
            sliced_seq = seq[:, :, -shift:]
            return torch.cat([sliced_seq, pad], dim=2)


class MyDynamicAvgPool1d(nn.Module):
    def __init__(self):
        """
        Initializes the EfficientDynamicAvgPool1d layer.
        """
        super(MyDynamicAvgPool1d, self).__init__()

    def forward(self, x, lengths=None):
        """
        Forward pass of the EfficientDynamicAvgPool1d layer.
        
        Args:
            x (Tensor): The input tensor of shape :math:`(N, C, L_{in})`.
            lengths (Tensor): A tensor of shape :math:`(N,)` indicating the length up to which to average for each sample.
        
        Returns:
            Tensor: The output tensor of shape :math:`(N, C)`.
        """
        # Create a mask based on lengths
        if lengths != None:
            max_length = x.size(2)
            
            # lengths must be smaller than max_length
            lengths = torch.clamp(lengths, max= max_length)
            # Create a mask based on lengths
            range_tensor = torch.arange(max_length, device=x.device).expand(len(lengths), max_length)
            lengths_expanded = lengths.unsqueeze(1).expand_as(range_tensor)
            # compare lengths with range_tensor to create a mask per position
            mask = range_tensor < lengths_expanded

            # Use the mask to zero out elements beyond the specified lengths
            masked_x = x * mask.unsqueeze(1).float()  # Mask applied across the channel dimension

            # Compute the sum along the length dimension and divide by the lengths to get the mean
            sums = masked_x.sum(dim=2)
            means = sums / lengths.float().unsqueeze(1)  # Ensure division is done in float
        else:
            means = x.mean(dim=2)

        return means

class AdamLH(Optimizer):
    """ AdamW with fully decoupled weight decay.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        self._init_lr = lr
        super(AdamLH, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamLH, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Parameters
        ----------
        closure : LossClosure, optional
            A callable that evaluates the model (possibly with backprop) and returns the loss,
            by default None.
        
        loss : torch.tensor, optional
            The loss tensor. Use this when the backward step has already been performed.
            By default None.
        

        Returns
        -------
        (Stochastic) Loss function value.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            lmbda = group['weight_decay']
            eps = group['eps']
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # decay
                p.mul_(1 - lmbda*lr/self._init_lr)

                grad = p.grad
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']


                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = lr / bias_correction1
                update = -step_size * exp_avg / denom
                p.add_(update)
                
        return loss
