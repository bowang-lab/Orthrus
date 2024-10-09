import torch
import torch.nn as nn
import torch.nn.functional as F
from orthrus.layers import StochasticShift


class LayerNormalization(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return F.layer_norm(x, x.size()[1:], eps=self.epsilon)


class SalukiConv1D(nn.Module):
    def __init__(
        self, 
        filters_in: int, 
        filters_out: int, 
        kernel_size: int, 
        initializer: str,
        dropout: float, 
        ln_epsilon: float, 
    ):
        super(SalukiConv1D, self).__init__()

        self.ln1 = LayerNormalization(epsilon=ln_epsilon)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv1d(
            in_channels=filters_in, 
            out_channels=filters_out, 
            kernel_size=kernel_size, 
            padding="valid"
        )
        self.d1 = nn.Dropout(dropout)

        self.max1 = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = self.ln1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.d1(x)
        x = self.max1(x)
        return x
    

def make_saluki_layer(
    num_layers: int, 
    filters_in: int, 
    filters_out: int, 
    kernel_size: int, 
    initializer: str="he_normal", 
    dropout: float=0.3, 
    ln_epsilon: float=0.007,
):
    layers = []
    for _ in range(num_layers):
        layers.append(
            SalukiConv1D(
                filters_in,
                filters_out,
                kernel_size,
                initializer,
                dropout,
                ln_epsilon,
            )
        )
    return nn.Sequential(*layers)


class SalukiModel(nn.Module):
    def __init__(
        self, 
        activation="relu", 
        seq_length=12288, 
        augment_shift=3, 
        heads=2,
        filters=64, 
        kernel_size=5, 
        dropout=0.3, 
        ln_epsilon=0.007, 
        num_layers=6,
        bn_momentum=0.90, 
        residual=False, 
        initializer="he_normal", 
        seq_depth=6, 
        go_backwards=True,
        final_layer=False,
        add_shift=True,
    ):
        super(SalukiModel, self).__init__()
        self.go_backwards = go_backwards
        if add_shift:
            self.shift = StochasticShift(
                shift_max=augment_shift, 
                symmetric=False
            )
        else:
            self.shift = nn.Identity()

        self.conv0 = nn.Conv1d(
            in_channels=seq_depth, 
            out_channels=filters, 
            kernel_size=kernel_size, 
            padding='valid', 
            bias=False
        )
        
        self.mid = make_saluki_layer(
            num_layers=num_layers,
            filters_in=filters,
            filters_out=filters,
            kernel_size=kernel_size,
            initializer=initializer,
            dropout=dropout,
            ln_epsilon=ln_epsilon,
        )
        
        self.ln = LayerNormalization(epsilon=ln_epsilon)

        self.rnn_layer = nn.GRU(
            input_size=filters, 
            hidden_size=filters, 
            batch_first=True, 
            bidirectional=False
        )
        self.bn2 = nn.BatchNorm1d(filters, momentum=bn_momentum)
        self.dense2 = nn.Linear(filters, filters)
        self.d2 = nn.Dropout(dropout)

        self.bn3 = nn.BatchNorm1d(filters, momentum=bn_momentum)
        if final_layer:
            self.final_fc = nn.Linear(filters, heads)
        else:
            self.final_fc = nn.Identity()

    def forward(self, inputs):
        x = self.shift(inputs)
        x = self.conv0(x)
        x = self.mid(x)
        x = self.ln(x)
        x = F.relu(x)

        # Input to GRU is B x L x C
        x = torch.transpose(x, 1, 2)
        
        # reflect the sequence along the time axis
        if self.go_backwards:
            x = torch.flip(x, [1])

        # get only the last state from the RNN
        _, x = self.rnn_layer(x)
        x = x.squeeze()
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = self.d2(x)

        x = self.bn3(x)
        x = F.relu(x)
        x = self.final_fc(x)

        return x

    def representation(self, inputs, lengths=None):
        x = self.forward(inputs)
        return x