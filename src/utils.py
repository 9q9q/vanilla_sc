"""Utils for sparse coding."""
import os
import urllib.request

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
import scipy.io as sio
from sklearn.feature_extraction import image
import torch
from torchvision.utils import make_grid
from tqdm import tqdm

SMOOTH_INTERVALS = 50


def visualize_patches(patches, title=""):
    """
    Given patches of images in the dataset, create a grid and display it.

    Parameters
    ----------
    patches : Tensor of (batch_size, pixels_per_patch).

    title : String; title of figure. Optional.
    """
    size = int(np.sqrt(patches.size(1)))
    batch_size = patches.size(0)
    img_grid = []
    for i in range(batch_size):
        img = torch.reshape(patches[i, :], (1, size, size))
        img_grid.append(img)

    out = make_grid(img_grid, padding=1, nrow=int(
        np.sqrt(batch_size)), pad_value=torch.min(patches))[0, :, :]
    display(out, bar=False, title=title)


def visualize_bases(bases, title):
    """
    Given basis vectors, create a grid and display it.

    Parameters
    ----------
    patches : Tensor of (N, num_bases).

    title : String; title of figure. Optional.
    """
    size = int(np.sqrt(bases.size(0)))
    grid = []
    for i in range(bases.size(0)):
        grid.append(torch.reshape(bases[:, i], (1, size, size)))
    out = make_grid(grid, padding=1, nrow=8, pad_value=-1)[0, :, :]
    display(out, bar=False, title=title)


# def visualize_patches(imgs, recons):
#     size = int(np.sqrt(imgs.size(1)))
#     batch_size = imgs.size(0)
#     img_grid = []
#     recon_grid = []
#     # min = torch.inf
#     # max = 0
#     for i in range(batch_size):
#         img = torch.reshape(imgs[i, :], (1, size, size))
#         img_grid.append(img)
#         recon = torch.reshape(recons[i, :], (1, size, size))
#         recon_grid.append(recon)

#     #     if torch.max(img) > max:
#     #         max = torch.max(img)
#     #     if torch.max(recon) > max:
#     #         max = torch.max(recon)
#     #     if torch.min(img) < min:
#     #         min = torch.min(img)
#     #     if torch.min(recon) < min:
#     #         min = torch.min(recon)

#     # value_range = (min, max)
#     return (make_grid(img_grid, padding=1, nrow=int(np.sqrt(batch_size)), pad_value=torch.min(imgs))[0, :, :],
#             make_grid(recon_grid, padding=1, nrow=int(np.sqrt(batch_size)), pad_value=torch.min(recons))[0, :, :])


# def visualize_bases(bases):
#     size = int(np.sqrt(bases.size(0)))
#     grid = []
#     for i in range(bases.size(0)):
#         grid.append(torch.reshape(bases[:, i], (1, size, size)))
#     return make_grid(grid, padding=1, nrow=8, pad_value=-1)[0, :, :]


def display(img, title=None, bar=True, cmap="gray", dpi=150, vrange=None):
    """Display images in Jupyter notebook.

    Parameters
    ----------
    img : Tensor or NumPy array of image values to display.

    title : String; title of figure. Optional.

    bar : Whether to show color bar on the side. Optional; default True.

    cmap : Color map. Optional; default 'gray'.

    dpi : Controls size. Optional; default 150.

    vrange: Tuple (value_min, value_max). Optional (will set automatically).
    """
    plt.axis("off")
    if vrange:
        vmin = vrange[0]
        vmax = vrange[1]
    else:
        vmin = torch.min(img)
        vmax = torch.max(img)
    plt.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
    fig = plt.gcf()
    fig.set_dpi(dpi)
    if bar:
        plt.colorbar()
    if title:
        plt.title(title)
    plt.show()

# def display(img, title=None, bar=True):
#     """Display maybe nicer image in Jupyter notebook."""
#     plt.axis("off")
#     plt.imshow(img, vmin=torch.min(img), vmax=torch.max(img))
#     fig = plt.gcf()
#     fig.set_dpi(150)
#     if bar:
#         plt.colorbar()
#     if title:
#         plt.title(title)
#     plt.show()


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

# TODO write this better
# def extract_patches(imgs, patch_size, batch_size, seed=0):
#     """From natural images, extract batches of patches.

#     Parameters
#     ----------
#     imgs : Tensor of (height, width, num_imgs).

#     patch_size : Height or width of patch (assume square).

#     batch_size : Number of images in batch.

#     seed : Optional integer.

#     Returns
#     -------
#     patches : Tensor of dim (batch_size, patch_size*patch_size).
#     """
#     torch.manual_seed(seed)

#     img_size = imgs.shape[0]
#     img_idx = torch.randint(low=0, high=imgs.shape[2], size=(batch_size,))
#     batch = imgs[:, :, img_idx]

#     # get random upper left coord of patch
#     start = torch.randint(low=0, high=img_size -
#                           patch_size, size=(2, batch_size))

#     # get batch_size random 8x8 patches
#     patches = np.zeros((batch_size, patch_size, patch_size))
#     for i in range(batch_size):
#         patches[i, :, :] = batch[
#             start[0, i]:start[0, i]+patch_size, start[1, i]:start[1, i]+patch_size, i]

#     return patches.reshape((batch_size, patch_size*patch_size))


def visualize_imgs(imgs):
    """
    Visualize whitened images in a perceptually-friendly way.

    Parameters
    ----------
    imgs : Tensor of (height, width, num_imgs).
    """
    grid = []
    for i in range(imgs.shape[-1]):
        reshaped = torch.reshape(torch.Tensor(imgs[:, :, i]), (1, 512, 512))
        grid.append(reshaped)
    out = make_grid(grid, padding=10, nrow=5, pad_value=-5)[0, :, :]
    display(out, bar=False, dpi=200, vrange=(-2, 2))


def get_patches(imgs, epochs, batch_size, N, patch_dir, rng):
    """Preprocess or load patches for faster training, ~100MB.

    Parameters
    ----------
    imgs : Tensor of (height, width, num_imgs).

    epochs : Number of training epochs.

    batch_size : Batch size.

    N : Number of basis functions.

    patch_dir : Path at which to store patches.

    Returns
    -------
    patches : Tensor of (batch_size, pixels_per_patch).
    """
    patch_size = int(np.sqrt(N))
    if patch_dir and os.path.exists(patch_dir):
        print("patches already preprocessed at '{}'; loading".format(patch_dir))
        patches = torch.load(patch_dir)
    else:
        print("preprocessing patches; this will take a few minutes")
        patches = torch.zeros((epochs, batch_size, N))
        for i in tqdm(range(epochs)):
            patches[i, :, :] = torch.Tensor(
                extract_patches(imgs, patch_size, batch_size, rng))
        torch.save(patches, patch_dir)
    return patches


def maybe_download_data(img_path):
    """If whitened images have not been downloaded, download (~20MB).

    Returns
    -------
    imgs : Tensor of (height, width, num_imgs).
    """
    if not os.path.exists(img_path):
        print("downloading data")
        data_url = "http://rctn.org/bruno/data/IMAGES.mat"
        urllib.request.urlretrieve(data_url, img_path)
    else:
        print("data exists; not downloading")
    # 512x512x10 (10 512x512 whitened images)
    imgs = sio.loadmat(img_path)["IMAGES"]
    return imgs


def plot_loss(y, title):
    """Plots loss per epoch with smoothing."""
    x = np.arange(len(y))
    spline = make_interp_spline(x, y)
    x_ = np.linspace(x.min(), x.max(), SMOOTH_INTERVALS)
    y_ = spline(x_)
    plt.plot(x_, y_)
    plt.title(title)
    plt.show()


def plot_coeffs(coeffs):
    """Plots coefficients for one image patch in dataset."""
    plt.stem(coeffs)
    plt.title("patch coefficients")
    plt.show()
