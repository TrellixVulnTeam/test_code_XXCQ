"""

本代码是自行实现的基于 Pokemon 数据集的多分类问题。特点：
- 自定义数据集 Pokemon，包括 5 labels，每个类别有 200-300 张图片，一共 1196 张。
- 使用自行实现的自定义的 DataLoader；
- 网络基于自行实现的完整的 ResNet18，或 torchvision 已经训练好的 ResNet18。
- 完整的 training, validation, testing 流程；
- 包括 model checkpoints saving and loading；


实验结果：
1) 如果使用自行实现的 ResNet18 从头开始训练，validation/testing accuracy 最高都可达 88% 左右 
    (Testing correct: 207/233)。无任何特别优化。考虑到 Pokemon 只有 1196 张图片，其中只有 60% 
    用于 training 的话，这个初始准确率已经很不错了。
2) 如果使用 torchvision 提供的已经训练好的 ResNet18（即 Transfer Learning），则 validation/testing 
    accuracy 会提升到 95%（217/233）。相当不错。

"""

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
from torchvision.models import resnet18 as resnet18_pretrained


def evaluate(model, device, criteon, loader):
    """
    Evaluation function. 
    """
    correct = 0
    loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
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
                        default='If not empty, this code will draw images, loss curves, etc on tensorboard')
    parser.add_argument('--save_models_path', type=str,
                        default='./checkpoints/pokemon_models')
    parser.add_argument('--use_pretrained_model', action='store_true',
                        help='Use pretrained ResNet18 model from torchvision')
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
            label_idx = int(sample_labels[i].cpu())
            label_str += train_db.label2name[label_idx] + ' '
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
    if args.use_pretrained_model:
        # Transfer learning: use pretrained model provided by torchvision
        trained_model = resnet18_pretrained(pretrained=True)
        net = nn.Sequential(
            # Use all layers of pretrained model except last layer, since we need to solve our problem whose number
            # of classes is specific.
            # Here:
            # - .children() to get all layers;
            # - [:-1] to use all layers except last one;
            # - *list() to spread list to independent parameters, which are exactly the input parameters for nn.Sequential()
            *list(trained_model.children())[:-1],
            # Flatten the output: [b, 512, 1, 1] => [b, 512]
            nn.Flatten(start_dim=1),
            # [b, 512] => [b, num_class]
            nn.Linear(512, num_class)
        ).to(device)
    else:
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
                    100. * batch_idx * len(images) / train_size, loss.item()))

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
        filename = 'model-epoch-{}-valloss-{:.3f}-accuracy-{:.3f}.mdl'.format(
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
