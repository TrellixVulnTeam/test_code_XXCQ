import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import argparse

from lenet5 import LeNet5

from torch.utils.tensorboard import SummaryWriter

batchsz = 256


# Define a helper function to show images
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
        img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def render_sample_images(batch_images, writer, one_channel=False):
    # create grid of images
    img_grid = torchvision.utils.make_grid(batch_images, nrow=8)

    # show images
    matplotlib_imshow(img_grid, one_channel=one_channel)

    # write to tensorboard
    writer.add_image('mnist_images_sample_first_batch', img_grid)
    writer.close()


def plot_classes_preds(images, targets, preds, classes, image_num=4, rows=2):
    fig = plt.figure(figsize=(12, 24))
    randIdx = np.random.randint(len(images), size=image_num)
    cols = int(image_num / rows)
    for idx in range(image_num):
        ax = fig.add_subplot(rows, cols, idx + 1, xticks=[], yticks=[])
        img_idx = randIdx[idx]
        matplotlib_imshow(images[img_idx], one_channel=False)
        ax.set_title("GT: {0}\PD: {1}".format(classes[targets[img_idx]],
                                                classes[preds[img_idx]]),
                     color=("green" if preds[img_idx]
                            == targets[img_idx].item() else "red"))
    return fig


def main():
    parser = argparse.ArgumentParser(description='Test_setting')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epoch number')
    parser.add_argument('--tensorboard_window_name', type=str,
                        default='')
    args = parser.parse_args()

    # print(torch.cuda.is_available())

    # --- Create dataloader from CIFAR data
    # Here torchvision already has CIFAR10 dataset and provides API.
    # The second parameter 'True' or 'False' is to create training or testing dataset.
    cifar_train_dataset = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(
        cifar_train_dataset, batch_size=batchsz, shuffle=True)
    cifar_test_dataset = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(
        cifar_test_dataset, batch_size=batchsz, shuffle=True)

    # --- Verify dataset
    cifar_samples = DataLoader(
        datasets.CIFAR10('cifar', False, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]), download=True), batch_size=batchsz, shuffle=False)

    flag_tb_viewer = False
    if args.tensorboard_window_name:
        flag_tb_viewer = True
    if flag_tb_viewer:
        classes = cifar_test_dataset.classes
        sample_source, sample_target = iter(cifar_samples).next()
        print(classes)
        # print(source.shape, target.shape)
        writer = SummaryWriter('runs/' + args.tensorboard_window_name)
        render_sample_images(sample_source, writer)

    # --- Start model training and testing
    device = torch.device('cuda:1')
    net = LeNet5().to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    epochs = args.epochs

    for epoch in range(epochs):

        # NOTE to use .train() to switch to training mode
        net.train()
        for batch_idx, (data, target) in enumerate(cifar_train):
            data, target = data.to(device), target.to(device)

            logits = net(data)
            loss = criteon(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(cifar_train.dataset),
                    100. * batch_idx / len(cifar_train), loss.item()))

                if flag_tb_viewer:
                    # ...log the running loss
                    writer.add_scalar('train_loss_batch', loss.item(),
                                      epoch * len(cifar_train) + batch_idx)

        train_loss = loss.item()

        # NOTE to use eval() and check 'torch.no_grad()' to switch to testing mode.
        net.eval()
        with torch.no_grad():
            correct = 0
            test_loss = 0
            for batch_idx, (data, target) in enumerate(cifar_test):
                data, target = data.to(device), target.to(device)

                logits = net(data)
                test_loss += criteon(logits, target)

                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum()

            accuracy = 100. * float(correct) / len(cifar_test.dataset)
            print('Epoch: {}, testing loss: {}, accuracy: {:.2f}%\n'.format(
                epoch, test_loss, accuracy))

        if flag_tb_viewer:
            # Add losses and accuracy per epoch in seperate curves
            writer.add_scalar('train_loss_epoch', train_loss, epoch)
            writer.add_scalar('test_loss_epoch', test_loss, epoch)
            writer.add_scalar('test_accuracy_epoch', accuracy, epoch)

            # View some images
            data = sample_source.to(device)
            target = sample_target.to(device)
            logits = net(data)
            pred = logits.argmax(dim=1)
            writer.add_figure('Training_image_samples_new',
                              plot_classes_preds(data.cpu(),
                                                 target.cpu(),
                                                 pred.cpu(),
                                                 classes,
                                                 image_num=64,
                                                 rows=8),
                              global_step=epoch * len(cifar_train) +
                              batch_idx)

    if flag_tb_viewer:
        writer.close()


if __name__ == '__main__':
    main()
