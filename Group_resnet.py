'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mixed_Pooling(nn.Module):
    '''
    mixed pooling 
    '''
    def __init__(self):
        super(Mixed_Pooling, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.drop = nn.Dropout2d()
        
    def forward(self, x):
        GAP = self.gap(x)
        MP = self.mp(x)
        MG = torch.cat((GAP, MP), 1)
        
        # mixed 
        out = self.drop(MG)
        
        return out
        
class DPBlock(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=7):
        super(DPBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=7, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out
    
class SeBlock(nn.Module):
    '''adaptive sequeeze convolutional layer'''
    def __init__(self, in_planes, out_planes):
        super(SeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, 32, kernel_size= 1, stride=1, padding=0, bias = False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, out_planes,
                                kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        
        return out
    
    
    
class AFSBlock(nn.Module):
    
    def __init__(self, in_channels, group = 2):
        '''
        :in_channels, the number of input channels
        : 
        '''
        super(AFSBlock,self).__init__()
        self.nchannels = in_channels
        self.groups = group
        
        self.split_channels = [in_channels // self.groups for _ in range(self.groups)]
        self.split_channels[0] +=in_channels -sum(self.split_channels)
        
        # group 1 GAP
        self.se_group_1 = SeBlock(in_planes= self.split_channels[0], out_planes=3)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        
        # group2 
        self.se_group_2 = SeBlock(in_planes= self.split_channels[1], out_planes=3)
        
        self.dp_pool = DPBlock(3,3)   
        
        # using depthwise convolution and 
       
    def forward(self, x):  
        # input tensor shape
        b,c,h,w = x.size()
        split_x = torch.split(x, self.split_channels, dim=1)
        
        se_group_1 = self.se_group_1(split_x[0])
        
        pooling_1 = self.gap(se_group_1)
        #print(pooling_1.size())
        se_group_2 = self.se_group_2(split_x[1])
        
        pooling_2 = self.gap(se_group_2)
        
        #print(pooling_2.size())
        pooling = torch.cat((pooling_1, pooling_2), 1)
        
        return pooling


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(6, num_classes)
        self.avg = AFSBlock(512,2)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def G_ResNet18(num_classes=3):
    return ResNet(BasicBlock, [2,2,2,2],num_classes)

def G_ResNet34(num_classes=3):
    return ResNet(BasicBlock, [3,4,6,3],num_classes)

def G_ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3],num_classes)

def G_ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3],num_classes)

def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3],num_classes)


def test():
    net = G_ResNet18(num_classes=3)
    y = net(torch.randn(1,3,32,32))
    print(y.size())
    
if __name__ == '__main__':
    test()