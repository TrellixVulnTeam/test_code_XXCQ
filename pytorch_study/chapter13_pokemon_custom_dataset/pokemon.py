"""

Create a custom dataset for Pokemon dataset (5 folders, 1165 images).

"""

import torch
import os
import glob
import random
import csv

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class Pokemon(Dataset):
    """
    Pokemon dataset contains 5 types of spirits saved in 5 different folders.
    Each folder contains 200-300 images. Total image number: 1165.
    """

    def __init__(self, root, resize_resolution, mode):
        super(Pokemon, self).__init__()

        self.root = root
        self.resize = resize_resolution

        # Raw Pokemon dataset only contains images saved in 5 folders, no labels.
        # So here we use each folder's index (0,1,2,3,4) as groundtruth labels.
        print('root_path:', root)
        self.name2label = {}
        self.label2name = {}
        for name in os.listdir(root):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
            self.label2name[len(self.label2name.keys())] = name
        # print(self.name2label)
        self.images, self.labels = self.load_csv('images.csv')

        # Create image and label list for all dataset types
        image_num = len(self.images)
        if mode == 'train':
            # Use first 60% for training
            self.images = self.images[:int(0.6 * image_num)]
            self.labels = self.labels[:int(0.6 * image_num)]
        elif mode == 'val':
            # Use 20% for validation (60% - 80% part)
            self.images = self.images[int(
                0.6 * image_num):int(0.8 * image_num)]
            self.labels = self.labels[int(
                0.6 * image_num):int(0.8 * image_num)]
        else:
            # Use the last 20% for testing (80% - 100% part)
            self.images = self.images[int(0.8 * image_num):]
            self.labels = self.labels[int(0.8 * image_num):]

        self.target_mean = [0.485, 0.456, 0.406]
        self.target_std = [0.229, 0.224, 0.225]

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            # If csv file is not found, create a new one.
            # Find all images in the pokemon dataset
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))

            print(len(images))
            # print(images)

            # Create a new csv file and put image paths in
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  # example: 'pokemon/bulbasaur/00000000.png'
                    # Get folder name, like 'bulbasaur'
                    name = img.split(os.path.sep)[-2]
                    label = self.name2label[name]
                    # 'pokemon/bulbasaur/00000000.png', 0
                    writer.writerow([img, label])
                print('writen into csv file:', filename)

        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'pokemon/bulbasaur/00000000.png', 0
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # idx: [0, len(images))
        image, label = self.images[idx], self.labels[idx]

        # Add transformer on image.
        # 它是串行的，可以任意放多个不同的变换形式，也支持自定义的函数（如下面的第一个函数）
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # image path -> image data
            # resize original image slightly larger (and crop it later)
            transforms.Resize(
                (int(self.resize * 1.25), int(self.resize * 1.25))),
            # add a small rotation for data augmentation
            transforms.RandomRotation(15),
            # crop to target image size
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),  # put image to tensor
            transforms.Normalize(mean=self.target_mean,
                                 std=self.target_std)  # image normalization on all channels
        ])
        image = tf(image)
        label = torch.tensor(label)

        return image, label

    def denormalize(self, x_hat):
        """
        Reverse image normalization step to recover unnormalized image. This is for 
        better image rendering, since normalized images look unrealistic for rendering.
        """
        # [3] => [3,1,1]
        mean = torch.tensor(self.target_mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(self.target_std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)

        # x_hat: [3, w, w], add [3, 1, 1] is fine, while add [3] is ERROR.
        x = x_hat * std + mean

        return x


def main():
    db = Pokemon('pokemon', 224, 'train')
    # db.load_csv('pokemon.csv')

    image, label = next(iter(db))

    print(image.shape, type(image))
    print(label)


if __name__ == '__main__':
    main()
