import torch.nn as nn
import torch.nn.functional as F
import torch

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


class SequenceProjectionHead(nn.Module):
    def __init__(
        self,
        input_features: int,
        projection_body: int,
        projection_head_size: int,
        norm_type: str
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_features, projection_body)
        self.norm1 = return_norm_layer(norm_type, projection_body)
        self.fc2 = nn.Linear(projection_body, projection_body)
        self.norm2 = return_norm_layer(norm_type, projection_body)
        self.fc3 = nn.Linear(projection_body, projection_head_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = x.permute(0, 2, 1)
        x = F.relu(x)

        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        x = self.norm2(x)
        x = x.permute(0, 2, 1)
        x = F.relu(x)
        x = self.fc3(x)
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
