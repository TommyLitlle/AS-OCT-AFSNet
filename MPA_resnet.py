"""SE-ResNet in PyTorch
Based on preact_resnet.py

Author: Xu Ma.
Date: Apr/15/2019
"""
import torch
import torch.nn as nn
import torch.nn.functional as F



class MGPooling(nn.Module):
    def __init__(self, nchannels, reduction = 16):
        super(MGPooling,self).__init__()
        self.nchannels = nchannels
        #self.groups = 2

        #self.split_channels = [nchannels // self.groups for _ in range(self.groups)]
        #self.split_channels[0] += nchannels - sum(self.split_channels)

        
        self.avg_conv = nn.Conv2d(in_channels=self.nchannels,out_channels=self.nchannels,
                                         kernel_size= 1, stride=1, padding=0, bias = False)
        self.max_conv = nn.Conv2d(in_channels=self.nchannels,out_channels=self.nchannels,
                                         kernel_size= 1, stride=1, padding=0, bias = False)
            
        self.avg_pool= nn.AdaptiveAvgPool2d(1) 
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool2 = nn.AdaptiveMaxPool2d(2)
        self.max_pool4 = nn.AdaptiveMaxPool2d(4)
        
        self.fc = nn.Sequential(
            nn.Linear(nchannels*10, nchannels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nchannels // reduction, nchannels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # input tensor shape
        b, c, _, _ = x.size()
        
        #split_x = torch.split(x, self.split_channels, dim=1)
        #avg_conv = self.avg_conv(x)
        #max_conv = self.max_conv(x)                                            
        
        # different pooling methods 
        avg_pool = self.avg_pool(x).view(b, c)
        avg_pool2 = self.avg_pool2(x).view(b, 4*c)
        #avg_pool4 = self.avg_pool4(x)
        
      
        max_pool = self.max_pool(x).view(b, c)
        max_pool2 = self.max_pool2(x).view(b, 4*c)
        #max_pool4 = self.max_pool4(x)
        #channel_1 = self.split_channels[0]
        #channel_2 = self.split_channels[1]
        
        #y1 = avg_pool.view(b, c)
        #y2 = max_pool.view(b, c)

        y = torch.cat((avg_pool, avg_pool2, max_pool, max_pool2), 1)
        y = self.fc(y).view(b, c, 1, 1)
    
    
        return x * y.expand_as(x)
    
    


class SPPSEPreActBlock(nn.Module):
    """SE pre-activation of the BasicBlock"""
    expansion = 1 # last_block_channel/first_block_channel

    def __init__(self,in_planes,planes,stride=1,reduction=16):
        super(SPPSEPreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=True)
        self.se = MGPooling(planes,reduction)
        if stride !=1 or in_planes!=self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,self.expansion*planes,kernel_size=1,stride=stride,bias=True)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        # Add SE block
        out = self.se(out)
        out += shortcut
        return out


class SPPSEPreActBootleneck(nn.Module):
    """Pre-activation version of the bottleneck module"""
    expansion = 4 # last_block_channel/first_block_channel

    def __init__(self,in_planes,planes,stride=1,reduction=16):
        super(SPPSEPreActBootleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,self.expansion*planes,kernel_size=1,bias=False)
        self.se = MGPooling(self.expansion*planes, reduction)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        # Add SE block
        out = self.se(out)
        out +=shortcut
        return out


class SPPSEResNet(nn.Module):
    def __init__(self,block,num_blocks,num_classes=3,reduction=16):
        super(SPPSEResNet, self).__init__()
        self.in_planes=64
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,reduction=reduction)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,reduction=reduction)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,reduction=reduction)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,reduction=reduction)
        self.avg_pol2d = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    #block means SEPreActBlock or SEPreActBootleneck
    def _make_layer(self,block, planes, num_blocks,stride,reduction):
        strides = [stride] + [1]*(num_blocks-1) # like [1,1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride,reduction))
            self.in_planes = planes*block.expansion
            
            
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out =  self.avg_pol2d(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PymaridResNet18(num_classes=3):
    return SPPSEResNet(SPPSEPreActBlock, [2,2,2,2],num_classes)


def PymaridResNet34(num_classes=3):
    return SPPSEResNet(SPPSEPreActBlock, [3,4,6,3],num_classes)


def PymaridResNet50(num_classes=3):
    return SPPSEResNet(SPPSEPreActBootleneck, [3,4,6,3],num_classes)


def PymaridResNet101(num_classes=3):
    return SPPSEResNet(SPPSEPreActBootleneck, [3,4,23,3],num_classes)


def PymaridResNet152(num_classes=10):
    return SPPSEResNet(SPPSEPreActBootleneck, [3,8,36,3],num_classes)


def test():
    net = PymaridResNet18()
    y = net((torch.randn(1,3,224, 224)))
    print(y.size())

# test()


