

import argparse
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from pokemon import Pokemon
from tensorboard_utils import render_batch_images
from torch.utils.tensorboard import SummaryWriter
from resnet18 import ResNet18
from torch import nn, optim


def evaluate(model, device, criteon, loader):
    correct = 0
    loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for x, y in enumerate(loader):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss += criteon(logits, y).item()

            pred = logits.argmax(dim=1)
            correct += pred.eq(y).sum()

        accuracy = float(correct) / len(loader.dataset)
    return accuracy, loss, correct


def main():
    parser = argparse.ArgumentParser(description='Test_setting')
    parser.add_argument('--root_path', type=str, default='pokemon',
                        help='root path for Pokemon dataset')
    parser.add_argument('--resize_image_resolution', type=int, default=224,
                        help='target image resolution after resizing')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epoch number')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training epoch number')
    parser.add_argument('--tensorboard_window_name', type=str,
                        default='')
    parser.add_argument('--save_models_path', type=str,
                        default='./checkpoints/pokemon_models')

    args = parser.parse_args()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    # -- Create dataset loader from our Pokemon class
    train_db = Pokemon(
        args.root_path, args.resize_image_resolution, mode='train')
    val_db = Pokemon(args.root_path, args.resize_image_resolution, mode='val')
    test_db = Pokemon(
        args.root_path, args.resize_image_resolution, mode='test')
    # 'num_workers' is to use multi-thread/process to load data.
    train_loader = DataLoader(train_db, batch_size=args.batch_size, shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(val_db, batch_size=args.batch_size, num_workers=2)
    test_loader = DataLoader(
        test_db, batch_size=args.batch_size, num_workers=2)

    # -- (Optional) If your dataset's structure is like the Pokemon dataset, that is,
    # all images are stored in several folders, then you can use torchvision API to load
    # all images. This will be short and easy. However, you still need to define your way
    # to load other information, like groundtruth labels, etc. This can be an option, but
    # actually not that convenient as our custom class.
    # tf = transforms.Compose([
    #                 transforms.Resize((64,64)),
    #                 transforms.ToTensor(),
    # ])
    # Here 'db' loads all images.
    # db = torchvision.datasets.ImageFolder(root=args.root_path, transform=tf)
    # loader = DataLoader(db, batch_size=32, shuffle=True)

    flag_tb_viewer = False
    if args.tensorboard_window_name:
        flag_tb_viewer = True
        writer = SummaryWriter('runs/' + args.tensorboard_window_name)

        sample_images, sample_labels = next(iter(train_loader))
        print('sample_images: {}, sample_labels: {}'.format(
            sample_images.shape, sample_labels.shape))

        # -- Render some image samples
        lst = []
        label_str = ''
        column_num = 4
        for i in range(0, 16):
            # Recover original images first (by denormalization)
            x = train_db.denormalize(sample_images[i])
            # print(x.shape)
            lst.append(x)

            # Add groundtruth labels as a string and render it on tensorboard too.
            label_str += train_db.label2name[sample_labels[i].cpu()] + ' '
            # Tensorboard text uses the markdown format (though it doesn't support all its
            # features). That means you need to add 2 spaces before \n to produce a linebreak,
            # e.g. line_1  \nline_2  \nline_3
            if (i+1) % column_num == 0:
                # There is already one space after each word (above). So add the second space here before \n.
                label_str += ' \n'
        batch_images = torch.stack(lst, dim=0)
        print(batch_images.shape)

        # Render one image
        writer.add_image('single_image', lst[0])
        # Render multiple images in grid
        render_batch_images(batch_images, writer, column_num=column_num)
        # Render string (GT labels)
        writer.add_text('sample_labels', label_str)

    # -- Define network and kick off training
    num_class = len(train_db.name2label.keys())
    net = ResNet18(num_class).to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)
    print('[STATS] train_size: {}, val_size: {}, test_size: {}'.format(
        train_size, val_size, test_size))

    best_accuracy = 0
    best_model_path = ''
    os.makedirs(args.save_models_path, exist_ok=True)

    for epoch in range(args.epochs):

        # -- Training
        net.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            logits = net(images)
            loss = criteon(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(images), train_size,
                    100. * batch_idx / train_size, loss.item()))

                if flag_tb_viewer:
                    # ...log the running loss
                    writer.add_scalar('train/loss/batch', loss.item(),
                                      epoch * train_size + batch_idx)

        train_loss = loss.item()

        # -- Validation. Put in a function, since it's the same as testing loop.
        val_accuracy, val_loss, correct = evaluate(
            net, device, criteon, val_loader)
        print('Epoch: {}, training loss: {}, validation loss: {}, validation correct: {}/{}, accuracy: {:.3f}%\n'.format(
            epoch, train_loss, val_loss, correct, val_size, val_accuracy))

        # -- Save models
        filename = 'model-epoch-{}-testloss-{:.3f}-accuracy-{:.3f}.mdl'.format(
            epoch, val_loss, val_accuracy)
        save_path = os.path.join(args.save_models_path, filename)
        torch.save(net.state_dict(), save_path)
        # Save the best model (with highest validation accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_path = os.path.join(
                args.save_models_path, 'model-best.mdl')
            torch.save(net.state_dict(), best_model_path)

        # -- Record curves on tensorboard
        if flag_tb_viewer:
            # Add losses and accuracy per epoch in seperate curves
            writer.add_scalar('train/loss/epoch', train_loss, epoch)
            writer.add_scalar('val/loss/epoch', val_loss, epoch)
            writer.add_scalar('val/accuracy', val_accuracy, epoch)

    # -- Testing step. Load the best validation model for testing.
    net.load_state_dict(torch.load(best_model_path))
    print('Load cpkt model:', best_model_path)

    test_accuracy, _, test_correct = evaluate(
        net, device, criteon, test_loader)
    print('Testing accuracy: {}, Testing correct: {}/{}'.format(test_accuracy,
          test_correct, len(test_loader.dataset)))

    if flag_tb_viewer:
        writer.close()


if __name__ == '__main__':
    main()
