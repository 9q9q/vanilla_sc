"""Utils for sparse coding."""

import os
import urllib.request

import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
from sklearn.feature_extraction import image
import torch
from torchvision.utils import make_grid


def visualize_patches(patches, title):
    size = int(np.sqrt(patches.size(1)))
    batch_size = patches.size(0)
    img_grid = []
    for i in range(batch_size):
        img = torch.reshape(patches[i, :], (1, size, size))
        img_grid.append(img)

    out = make_grid(img_grid, padding=1, nrow=int(np.sqrt(batch_size)), pad_value=torch.min(patches))[0,:,:]
    display(out, bar=False, title=title)


def visualize_bases(bases, title):
    size = int(np.sqrt(bases.size(0)))
    grid = []
    for i in range(bases.size(0)):
        grid.append(torch.reshape(bases[:, i], (1, size, size)))
    out = make_grid(grid, padding=1, nrow=8, pad_value=-1)[0, :, :]
    display(out, bar=False, title=title)


def visualize_imgs(imgs):
    grid = []
    for i in range(imgs.shape[-1]):
        grid.append(torch.reshape(torch.Tensor(imgs[:, :, i]), (1, 512, 512)))
    out = make_grid(grid, padding=10, nrow=5, pad_value=-5)[0,:,:]
    display(out, bar=False, dpi=200)


def display(img, title=None, bar=True, cmap="gray", dpi=150):
    """Display maybe nicer image in Jupyter notebook."""
    plt.axis("off")
    plt.imshow(img, vmin=torch.min(img), vmax=torch.max(img), cmap=cmap)
    fig = plt.gcf()
    fig.set_dpi(dpi)
    if bar:
        plt.colorbar()
    if title:
        plt.title(title)
    plt.show()


def plot_line(y, title):
    x_ax = np.arange(len(y))
    plt.plot(x_ax, y)
    plt.title(title)
    plt.show()


def plot_coeffs(coeffs):
    plt.stem(coeffs)
    plt.title("patch coefficients")
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


def maybe_download_data(img_path):
    if not os.path.exists(img_path):
        print("downloading data")
        data_url = "http://rctn.org/bruno/data/IMAGES.mat"
        urllib.request.urlretrieve(data_url, img_path)
    else:
        print("data exists; not downloading")
    imgs = sio.loadmat(img_path)["IMAGES"] # 512x512x10 (10 512x512 binary, whitened images)
    visualize_imgs(imgs)