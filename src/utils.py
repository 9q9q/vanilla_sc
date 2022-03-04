"""Utils for sparse coding."""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction import image
import torch
from torchvision.utils import make_grid


def visualize_patches(imgs, recons):
    size = int(np.sqrt(imgs.size(1)))
    batch_size = imgs.size(0)
    img_grid = []
    recon_grid = []
    # min = torch.inf
    # max = 0
    for i in range(batch_size):
        img = torch.reshape(imgs[i, :], (1, size, size))
        img_grid.append(img)
        recon = torch.reshape(recons[i, :], (1, size, size))
        recon_grid.append(recon)

    #     if torch.max(img) > max:
    #         max = torch.max(img)
    #     if torch.max(recon) > max:
    #         max = torch.max(recon)
    #     if torch.min(img) < min:
    #         min = torch.min(img)
    #     if torch.min(recon) < min:
    #         min = torch.min(recon)

    # value_range = (min, max)
    return (make_grid(img_grid, padding=1, nrow=int(np.sqrt(batch_size)), pad_value=torch.min(imgs))[0,:,:],
     make_grid(recon_grid, padding=1, nrow=int(np.sqrt(batch_size)), pad_value=torch.min(recons))[0,:,:])


def visualize_bases(bases):
    size = int(np.sqrt(bases.size(0)))
    grid = []
    for i in range(bases.size(0)):
        grid.append(torch.reshape(bases[:, i], (1, size, size)))
    return make_grid(grid, padding=1, nrow=8, pad_value=-1)[0, :, :]


def display(img, title=None, bar=True):
    """Display maybe nicer image in Jupyter notebook."""
    plt.axis("off")
    plt.imshow(img, vmin=torch.min(img), vmax=torch.max(img))
    fig = plt.gcf()
    fig.set_dpi(150)
    if bar:
        plt.colorbar()
    if title:
        plt.title(title)
    plt.show()


# TODO write this better
def extract_patches(imgs, patch_size, batch_size, rng):
    """Want 64xbs patch mtx, so one patch is 8x8 patch of 512.
    """
    img_size = imgs.shape[0]
    img_idx = rng.integers(low=0, high=imgs.shape[2], size=batch_size)
    batch = imgs[:, :, img_idx]

    # get random upper left coord of patch
    start = rng.integers(low=0, high=img_size -
                         patch_size, size=2*batch_size)
    start = start.reshape(2, batch_size)

    # get batch_size random 8x8 patches
    patches = np.zeros((batch_size, patch_size, patch_size))
    for i in range(batch_size):
        patches[i, :, :] = batch[
            start[0, i]:start[0, i]+patch_size, start[1, i]:start[1, i]+patch_size, i]

    return patches.reshape((batch_size, patch_size*patch_size))