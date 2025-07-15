import torch
import torch.nn as nn
import torch.nn.functional as F
from orthrus.layers import return_norm_layer, DilatedBasicBlockLayer, StochasticShift, MyDynamicAvgPool1d

class DilatedResnet(nn.Module):
    def __init__(
        self, 
        num_classes, 
        layer_params, 
        dilation_params=(1, 2, 4, 8), 
        dropout_prob=0, 
        pooling_layer='max_pool', 
        kernel_size=2, 
        filter_nums=(64, 128, 128, 256), 
        norm_type="batchnorm", 
        increase_dilation=False,
        add_shift=True,
        n_tracks=6,
        global_pooling_layer='avgpool',
        final_layer=False,
    ):
        super(DilatedResnet, self).__init__()
        self.num_classes = num_classes
        self.dilation_params = dilation_params
        self.dropout_prob = dropout_prob
        self.pooling_layer = pooling_layer
        self.kernel_size = kernel_size
        self.filter_nums = filter_nums
        self.norm_type = norm_type
        self.increase_dilation = increase_dilation
        self.global_pooling_layer = global_pooling_layer
        
        # Initial convolution
        if add_shift:
            self.shift = StochasticShift(3, symmetric=False)
        else:
            self.shift = nn.Identity()
        self.conv1 = nn.Conv1d(n_tracks, filter_nums[0], kernel_size=kernel_size, padding='same')
        self.bn1 = return_norm_layer(norm_type, filter_nums[0])
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        # Dilated blocks
        self.layers = nn.ModuleList()
        in_channels = [filter_nums[0]] + list(filter_nums[:-1])
        for i, (in_channel, filter_num, block, dilation) in enumerate(
            zip(in_channels, filter_nums, layer_params, dilation_params)
        ):
            self.layers.append(
                DilatedBasicBlockLayer(
                    in_channels=in_channel, 
                    filter_num=filter_num, 
                    kernel_size=kernel_size, 
                    padding='same', 
                    dropout_prob=dropout_prob if i < len(filter_nums) - 1 else 0, 
                    dilation=dilation, 
                    pooling_layer=pooling_layer, 
                    blocks=block, 
                    norm_type=norm_type, 
                    increase_dilation=increase_dilation,
                )
            )
            in_channels = filter_num
            
        # Adaptive pooling and classification layer
        if global_pooling_layer in ['avgpool', 'dynamic_avgpool']:
            self.avgpool = MyDynamicAvgPool1d()
        else:
            raise ValueError
        
        if final_layer:
            self.fc = nn.Linear(filter_nums[-1], num_classes)
        else:
            self.fc = nn.Identity()

    def forward(self, x, lengths=None):
        x = self.representation(x, lengths=lengths)
        x = self.fc(x)
        
        return x
    
    def un_pooled_representation(self, x):
        x = self.shift(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def representation(self, x, lengths=None):
        x = self.un_pooled_representation(x)
        
        x = self.avgpool(x, lengths=lengths)
        x = torch.flatten(x, 1)
        return x        
        
def dilated_small(
    num_classes,
    dropout_prob=0.1,
    norm_type="batchnorm",
    dilation_params=(1, 2, 4, 8),
    kernel_size=2,
    pooling_layer='max_pool',
    add_shift=True,
    increase_dilation=False,
    global_pooling_layer='avgpool',
    final_layer=False,
):
    return DilatedResnet(
        num_classes=num_classes,
        filter_nums=(64, 128, 128, 256),
        layer_params=(2, 2, 2, 2),
        dropout_prob=dropout_prob,
        dilation_params=dilation_params,
        norm_type=norm_type,
        kernel_size=kernel_size,
        pooling_layer=pooling_layer,
        increase_dilation=increase_dilation,
        add_shift=add_shift,
        global_pooling_layer=global_pooling_layer,
        final_layer=final_layer,
    )


def dilated_medium(
    num_classes,
    dropout_prob=0.1,
    norm_type="batchnorm",
    dilation_params=(1, 2, 4, 8),
    kernel_size=4,
    pooling_layer='avgpool',
    add_shift=True,
    increase_dilation=False,
    global_pooling_layer='avgpool',
    final_layer=False,
):
    return DilatedResnet(
        num_classes=num_classes,
        filter_nums=(64, 128, 256, 512),
        layer_params=(3, 4, 6, 3),
        dropout_prob=dropout_prob,
        dilation_params=dilation_params,
        norm_type=norm_type,
        kernel_size=kernel_size,
        pooling_layer=pooling_layer,
        increase_dilation=increase_dilation,
        add_shift=add_shift,
        global_pooling_layer=global_pooling_layer,
        final_layer=final_layer,
    )


def not_dilated_small(
    num_classes,
    dropout_prob=0.1,
    norm_type="batchnorm",
    dilation_params=(1, 1, 1, 1),
    kernel_size=2,
    pooling_layer='max_pool',
    add_shift=True,
    increase_dilation=False,
    global_pooling_layer='avgpool',
    final_layer=False,
):
    return DilatedResnet(
        num_classes=num_classes,
        filter_nums=(64, 128, 128, 256),
        layer_params=(2, 2, 2, 2),
        dropout_prob=dropout_prob,
        dilation_params=dilation_params,
        norm_type=norm_type,
        kernel_size=kernel_size,
        pooling_layer=pooling_layer,
        increase_dilation=increase_dilation,
        add_shift=add_shift,
        global_pooling_layer=global_pooling_layer,
        final_layer=final_layer,
    )
    