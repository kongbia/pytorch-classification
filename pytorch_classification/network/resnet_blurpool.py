'''
# Resnet for blur pool.
'''
import torch.nn as nn
import math
from .blur_pool import BlurPool


__all__ = ['resnet_blurpool']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1_bn_relu = nn.Sequential(conv3x3(inplanes, planes, 1), 
                                          nn.BatchNorm2d(planes),
                                          nn.ReLU(inplace=True))
        if stride > 1:
            self.conv1_bn_relu.add_module('blurpool',
                                            BlurPool(channels=planes, stride=stride))

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1_bn_relu(x)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2_bn_relu = nn.Sequential(
                                    nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                                padding=1, bias=False),\
                                    nn.BatchNorm2d(planes))
        if stride > 1:
            self.conv2_bn_relu.add_module('blurpool',
                                            BlurPool(channels=planes, stride=stride))

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2_bn_relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):
    def __init__(self, depth, num_classes=1000, block_name='BasicBlock', network_stride=(1, 1, 2, 2),):
        super(ResNet, self).__init__()
        assert len(network_stride) == 4
        assert all([s == 1 or s == 2 for s in network_stride])
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')


        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=network_stride[0],
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n, stride=network_stride[1])
        self.layer2 = self._make_layer(block, 32, n, stride=network_stride[2])
        self.layer3 = self._make_layer(block, 64, n, stride=network_stride[3])
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            if stride > 1:
                downsample.add_module('blurpool',
                    BlurPool(channels=planes * block.expansion, stride=stride)) 


        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def resnet_blurpool(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)


if __name__ == '__main__':
    import torch
    net = resnet_blurpool(block_name='BasicBlock', depth=20, num_classes=10).cuda()
    doc = open('blurpool.txt','w')
    print(net, file=doc)
    # print(net)

    input = torch.FloatTensor(4,3,32,32).cuda()

    # print(output.shape)
    print('    Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))