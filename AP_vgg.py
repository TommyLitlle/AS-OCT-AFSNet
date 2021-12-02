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
              torch.nn.init.constant_(m.weight, 1/49)
    def forward(self, x):
        out = self.conv1(x)
        
        return out
        
    
class AP_VGG(nn.Module):
    def __init__(self, vgg_name):
        super(AP_VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        
        self.classifier = nn.Linear(512, 3)

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
        
        layers += [ Dynamic_GAP(512)]
        #layers += [nn.AvgPool2d(kernel_size=16, stride=16)]
        return nn.Sequential(*layers)


def test():
    net = AP_VGG('VGG19')
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y.size())
    
if __name__ == '__main__':  
    test()
