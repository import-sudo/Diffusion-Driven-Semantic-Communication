import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

class ResidualUpBlock(nn.Module):
    def __init__(self, planes, mid_planes):
        super(ResidualUpBlock, self).__init__()

        self.conv1 = nn.Sequential(
                nn.Conv2d(planes, mid_planes, kernel_size=3, padding=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(mid_planes),
            )
        self.conv2 = nn.Sequential(
                nn.Conv2d(mid_planes, planes, kernel_size=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(planes),
            )                        

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        out += residual

        return out



class up_model(nn.Module):
    def __init__(self, layers=[2,2,3,4,5]):  # layers=参数列表 block选择不同的类
        super(up_model, self).__init__()
        self.mask = nn.Parameter(torch.ones(1, 512))

        self.fc = nn.Linear(512 , 7*7*32)
        self.conv1 = nn.Sequential(
                nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1,bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(128),
            )  

        self.conv2 = nn.Sequential(
                nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1,bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(512),
            ) 

        self.layer1 = self._make_layer(256, layers[0])
        self.layer2 = self._make_layer(128, layers[1])
        self.layer3 = self._make_layer(64, layers[2])
        self.layer4 = self._make_layer(32, layers[3])
        self.layer4 = self._make_layer(16, layers[4])

        self.conv3 = nn.Conv2d(16, 3, kernel_size=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        up_layer = nn.ConvTranspose2d(planes * 2, planes , kernel_size=2, stride=2)
        layers = [up_layer]

        for i in range(0, blocks):
            layers.append(ResidualUpBlock(planes, planes // 2))   #该部分是将每个blocks的剩下residual 结构保存在layers列表中，这样就完成了一个blocks的构造。

        return nn.Sequential(*layers)

    def forward(self, x, sparse=False):
        if sparse:
            x = x * self.mask

        x = self.fc(x) #7*7*32
        x = x.reshape(x.shape[0], -1, 7, 7) # [32,7,7]
        x = self.conv1(x) 
        x = self.conv2(x)  # [512,7,7]

        x = self.layer1(x) # [256,14,14]
        x = self.layer2(x) # [128,28,28]
        x = self.layer3(x) # [64, 56,56]
        x = self.layer4(x) # [32, 112, 112]
        x = self.layer5(x) # [16, 224, 224]

        out = self.conv3(x)
        return out, self.mask
