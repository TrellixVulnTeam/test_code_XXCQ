import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import padding


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


def main():

    net = LeNet5()
    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    print('lenet out:', out.shape)


if __name__ == '__main__':
    main()
