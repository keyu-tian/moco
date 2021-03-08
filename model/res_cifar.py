import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCifar(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, stride=1, padding=1
    (ii) removes pool1
    """
    
    def __init__(self, norm_layer, num_classes, img_ch):
        num_blocks = [2, 2, 2, 2]
        
        super(ResNetCifar, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(img_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2, norm_layer=norm_layer)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
    
    def _make_layer(self, planes, num_blocks, stride, norm_layer):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride, norm_layer))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet18(norm_layer=nn.BatchNorm2d, num_classes=10, img_ch=3):
    return ResNetCifar(norm_layer, num_classes, img_ch)
