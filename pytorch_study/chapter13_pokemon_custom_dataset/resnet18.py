"""
Implement ResNet18 by myself.

"""

import torch
from torch import nn
from torch.nn import functional as F


class ResBlk(nn.Module):
    """
    A standard ResNet block.
    """

    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(
            ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(
            ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        # This extra layer will be on x to ensure its result has exactly
        # the same shape of F(x) to ensure they can add together. See
        # details in forward().
        self.extra = nn.Sequential()
        if ch_in != ch_out:
            # Here kernel_size=1, padding=0 and input stride are designed
            # to ensure the result shape is same as the shape of result from
            # above two layers conv1 -> conv2.
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1,
                          stride=stride, padding=0),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        # NOTE: have to use F.relu() instead of nn.ReLU(), since nn.ReLU()
        # only runs in-place and cannot take input parameter.
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Here we want to add x + F(x) and then run ReLU, so we have to
        # use extra(x) to ensure they have exactly the same shape first.
        # This is why we write separate conv1, bn1, conv2, bn2 above.
        out = self.extra(x) + out
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet18 implementation. 
    Refer to figures in this link: https://www.jianshu.com/p/085f4c8256f1.

    NOTE that the input image dimension in original ResNet is designed to be 224 
    (x is [b, 3, 224, 224]). If not, the tensor shape in some intermediate layers 
    might be weird and sometimes may cause conflicts (not always). The comments
    in this code is based on input dimension 224.
    """

    def __init__(self, num_class):
        super(ResNet18, self).__init__()

        # [b, 3, 224, 224] => [b, 64, 56, 56]
        self.conv1 = nn.Sequential(
            # [b, 3, 224, 224] => [b, 64, 112, 112]
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # [b, 64, 112, 112] => [b, 64, 56, 56]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # -- 8 ResNet blocks.
        # [b, 64, 56, 56] => [b, 64, 56, 56]
        self.blk1 = ResBlk(64, 64, stride=1)
        # [b, 64, 56, 56] => [b, 64, 56, 56].
        # The first two blocks are exactly the same
        self.blk2 = ResBlk(64, 64, stride=1)
        # [b, 64, 56, 56] => [b, 128, 28, 28]
        self.blk3 = ResBlk(64, 128, stride=2)
        # [b, 128, 28, 28] => [b, 128, 28, 28]
        self.blk4 = ResBlk(128, 128, stride=1)
        # [b, 128, 28, 28] => [b, 256, 14, 14]
        self.blk5 = ResBlk(128, 256, stride=2)
        # [b, 256, 14, 14] => [b, 256, 14, 14]
        self.blk6 = ResBlk(256, 256, stride=1)
        # [b, 256, 14, 14] => [b, 512, 7, 7]
        self.blk7 = ResBlk(256, 512, stride=2)
        # [b, 512, 7, 7] => [b, 512, 7, 7]
        self.blk8 = ResBlk(512, 512, stride=1)

        # Adaptive Average pool: [b, 512, 7, 7] => [b, 512, 1, 1].
        # Our target is to flatten the input tensor to [b,512,1,1], so
        # using adaptive pool is good that we don't need to care about parameters
        # like stride, padding, etc.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Last linear layer: [b, 512] => [b, num_class]
        self.final = nn.Linear(512, num_class)

    def forward(self, x):
        """
        # [b, 3, 224, 224] => [b, num_class]
        """
        out = self.conv1(x)
        out = self.blk1(out)
        out = self.blk2(out)
        out = self.blk3(out)
        out = self.blk4(out)
        out = self.blk5(out)
        out = self.blk6(out)
        out = self.blk7(out)
        out = self.blk8(out)
        out = self.avgpool(out)

        # [b, 512, 1, 1] => [b, 512*1*1].
        # Flatten the tensor before final linear layer.
        out = out.view(out.size(0), -1)

        out = self.final(out)
        return out
