import torch
from torch import nn
from torch.nn import functional as F


class LeNet5(nn.Module):
    """
    for cifar10 dataset.
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        # NOTE: Here we changed the first Conv layer with 3-channel input instead of 1-channel
        # input in original LeNet5.
        self.conv_unit = nn.Sequential(
            # x: [b, 3, 32, 32] => [b, 6, 28, 28]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            # [b, 6, 28, 28] => [b, 6, 14, 14]
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # [b, 6, 14, 14] => [b, 16, 10, 10]
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            # [b, 16, 14, 14] => [b, 16, 5, 5]
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # [b, 16, 5, 5] => [b, 120, 1, 1]
            nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
        )

        self.fc_unit = nn.Sequential(
            nn.Linear(120, 84),
            # Add a new ReLU here (LeNet5 doesn't have ReLU at that time)
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        batchsz = x.size(0)
        x = self.conv_unit(x)
        x = x.view(batchsz, -1)
        logits = self.fc_unit(x)
        return logits


class ResBlk(nn.Module):
    """
    ResNet block for cifar10 dataset.
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
    A simplified ResNet18 version for CIFAR dataset. The reason is that, each CIFAR image 
    is only [3, 32, 32]. The resolution is far smaller than 224x224 in original ResNet paper.
    So here we simplify the ResNet18 by almost half.
    """

    def __init__(self):
        super(ResNet18, self).__init__()

        # [b, 3, 32, 32] => [b, 64, 16, 16]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        # [b, 64, 16, 16] => [b, 64, 16, 16]
        self.blk1 = ResBlk(64, 64, stride=1)
        # [b, 64, 16, 16] => [b, 128, 8, 8]
        self.blk2 = ResBlk(64, 128, stride=2)
        # [b, 128, 8, 8] => [b, 256, 4, 4]
        self.blk3 = ResBlk(128, 256, stride=2)
        # [b, 256, 4, 4] => [b, 512, 2, 2]
        self.blk4 = ResBlk(256, 512, stride=2)

        # [b, 512, 2, 2] => [b, 512, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # [b, 512] => [b, 10]
        self.final = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.blk1(out)
        out = self.blk2(out)
        out = self.blk3(out)
        out = self.blk4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.final(out)
        return out


def main():

    net = LeNet5()
    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    print('lenet out:', out.shape)

    model = ResNet18()
    out = model(tmp)
    print('ResNet out:', out.shape)


if __name__ == '__main__':
    main()
