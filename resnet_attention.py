import mindspore
import mindspore.nn as nn
from mindspore import ops as P
import mindspore.common.initializer as weight_init
from functools import partial
import mindspore.numpy as np
from mindspore import Tensor, context, Parameter,Model

class GlobalAvgPooling(nn.Cell):
    """
    global average pooling feature map.

    Args:
         mean (tuple): means for each channel.
    """
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.mean = P.ReduceMean(False)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x

class GlobalMaxPooling(nn.Cell):
    """
    global average pooling feature map.

    Args:
         mean (tuple): means for each channel.
    """
    def __init__(self):
        super(GlobalMaxPooling, self).__init__()
        self.mean = P.ReduceMax(False)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x
    
class ChannelAttention(nn.Cell):
    """
    ChannelAttention: Since each channel of the feature map is considered as a feature detector, it is meaningful
    for the channel to focus on the "what" of a given input image;In order to effectively calculate channel attention,
    the method of compressing the spatial dimension of input feature mapping is adopted.
    """
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool =  GlobalAvgPooling()
        self.max_pool =  GlobalMaxPooling()
        self.maxp= P.ReduceMax(True)

        self.concat = P.Concat(axis=1)
        self.cast_op = P.Cast()
        
        self.sum1=P.ReduceSum(keep_dims=False)
        self.fc1 = nn.Dense(channel, channel // 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(channel // 16, channel)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        b, c,h,w = x.shape
        avg_out = self.avg_pool(x).reshape(b,1,c)
       
        max_out = self.max_pool(x).reshape(b,1,c)
    
        y=self.concat((avg_out,max_out))
        #y = self.cast_op(out, x.dtype)
        y= self.fc1(y)
        y= self.relu(y)
        y = self.fc2(y)
        y=self.sum1(y,1)
       
        y = self.sigmoid(y)
        y = y.reshape(b, c, 1, 1)

        return x*y


class SpatialAttention(nn.Cell):
    """
    SpatialAttention: Different from the channel attention module, the spatial attention module focuses on the
    "where" of the information part as a supplement to the channel attention module.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size)
        self.concat = P.Concat(axis=1)
        self.sigmoid = nn.Sigmoid()
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.max_pool = P.ReduceMax(keep_dims=True)

    def construct(self, x):
        avg_out = self.reduce_mean(x, 1)
        max_out = self.max_pool(x, 1)
        y = self.concat((avg_out, max_out))
        y = self.conv1(y)
        y = self.sigmoid(y)

        return x*y


class CBAM(nn.Cell):
    def __init__(self, channel,  kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention(kernel_size)
    def construct(self, x):
        out = self.channel_attention(x) 
        out = self.spatial_attention(out) 
        return out


class SE(nn.Cell):
    """
    squeeze and excitation block.

    Args:
        channel (int): number of feature maps.
        reduction (int): weight.
    """
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()

        self.avg_pool = GlobalAvgPooling()
        self.fc1 = nn.Dense(channel, channel // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(channel // reduction, channel)
        self.sigmoid = nn.Sigmoid()
        
    def construct(self, x):
        b, c,h,w = x.shape
        
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.reshape(b, c, 1, 1)
        return x * y

class Bottleneck(nn.Cell):
    """
    Residual structure.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU()

        self.att = CBAM(out_channels * 4)
        self.dowmsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.att(out) 

        if self.dowmsample is not None:
            residual = self.dowmsample(residual)

        out = out+ residual
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    Overall network architecture.
    """
    def __init__(self, block, layers, num_classes=1000):
        self.in_channels = 64
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.flatten = nn.Flatten()
        self.Linear = nn.Dense(512 * block.expansion, num_classes)
       
        
        self.init_weights()
    
    
    def init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            
            layers.append(block(self.in_channels, out_channels))

        return nn.SequentialCell(*layers)


    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.Linear(x)
        
        return x



    


def resnet50(num_classes=1000):
    """
    Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model

if __name__ == "__main__":
    from mindspore import context
    from mindspore import dtype as mstype
    #import numpy as np

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    data = Tensor(np.ones([2, 3, 224, 224]), dtype=mstype.float32)
    model = resnet50()
    # 验证前向传播
    out = model(data)
    print(out.shape)
    #params = 0.
    
    #for name, param in model.parameters_and_names():
    	# 获取参数, 获取名字
        #params += np.prod(param.shape)
        #print(name)
    #print(params, 26604328)