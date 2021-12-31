import torchvision
import matplotlib.pyplot as plt
import numpy as np


def matplotlib_imshow(img, one_channel=False):
    """
    Define a helper function to show images
    """
    if one_channel:
        img = img.mean(dim=0)
        img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        # matplotlib can only run image whose size is [width, height, channels],
        # while our image data is [channels, width, height].
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def render_batch_images(batch_images, writer, one_channel=False, column_num=8):
    # create grid of images
    img_grid = torchvision.utils.make_grid(batch_images, nrow=column_num)

    # show images
    matplotlib_imshow(img_grid, one_channel=one_channel)

    # write to tensorboard
    writer.add_image('mnist_images_sample_first_batch', img_grid)
    writer.close()
