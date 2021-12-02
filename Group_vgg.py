'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [32, 64, 'M', 128, 128, 'M', 192, 192, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 'M'],
}

class Dynamic_GAP(nn.Module):
    def __init__(self, in_planes, stride=7):
        super(Dynamic_GAP, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=7, stride=stride, padding=1, groups=in_planes, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.constant_(m.weight, 1)
    def forward(self, x):
        out = self.conv1(x)
        
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
        self.conv1 = nn.Conv2d(in_planes, 512, kernel_size= 1, stride=1, padding=0, bias = False)
        self.bn1 = nn.BatchNorm2d(512)
        
        self.conv2 = nn.Conv2d(512, out_planes,
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
        self.groups = 2
        
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

 
    
class Group_VGG(nn.Module):
    def __init__(self, vgg_name):
        super(Group_VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        
        self.classifier = nn.Linear(6, 3)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        
        layers += [ AFSBlock(512,2)]
        #layers += [nn.AvgPool2d(kernel_size=16, stride=16)]
        return nn.Sequential(*layers)


def test():
    net = Group_VGG('VGG19')
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y.size())
    
if __name__ == '__main__':  
    test()
