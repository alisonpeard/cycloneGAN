import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# import geopandas as gpd

from .tf_utils import *

def discrete_colormap(data, nintervals, min=None, cmap="cividis", under='dimgrey'):
    cmap = getattr(mpl.cm, cmap)
    if min is not None:
        data = data[data >= min]
    bounds = np.quantile(data.ravel(), np.linspace(0, 1, nintervals))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cmap.set_under(under)
    return cmap, norm


def plot_generated_marginals(fake_data, start=0):
    print(f"Range: [{fake_data.min():.2f}, {fake_data.max():.2f}]")

    nrows = 10
    fig, axs = plt.subplots(3, nrows, layout='tight', figsize=(10, 3))

    for i, ax in enumerate(axs.ravel()[:nrows]):
        im = ax.imshow(fake_data[start + i, ..., 0], cmap='YlOrRd', vmin=0, vmax=1)
    axs[0, 0].set_ylabel('wind')

    for i, ax in enumerate(axs.ravel()[nrows: 2 * nrows]):
        im = ax.imshow(fake_data[start + i, ..., 1], cmap='YlOrRd', vmin=0, vmax=1)
    axs[1, 0].set_ylabel('wave')

    for i, ax in enumerate(axs.ravel()[2 * nrows:]):
        im = ax.imshow(fake_data[start + i, ..., 2], cmap='YlOrRd', vmin=0, vmax=1)
    axs[1, 0].set_ylabel('precipitation')

    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.suptitle('Generated marginals')

    return fig


def plot_sample_density(data, ax, sample_pixels=None, cmap='cividis'):
    """For generated quantiles."""
    h, w = data.shape[1:3]
    n = h * w

    if sample_pixels is None:
        sample_pixels_x = random.sample(range(n), 1)
        sample_pixels_y = random.sample(range(n), 1)
    else:
        assert sample_pixels[0] != sample_pixels[1]
        sample_pixels_x = [sample_pixels[0]]
        sample_pixels_y = [sample_pixels[1]]

    data_ravel = tf.reshape(data, [len(data), n])

    sample_x = tf.gather(data_ravel, sample_pixels_x, axis=1)
    sample_y = tf.gather(data_ravel, sample_pixels_y, axis=1)

    # frechet_x = -tf.math.log(1 - sample_x)
    # frechet_y = -tf.math.log(1 - sample_y)
    # ec_xy = raw_extremal_correlation(frechet_x, frechet_y)
    axtitle = f"Pixels ({sample_pixels_x[0]}, {sample_pixels_y[0]})"
    scatter_density(sample_x.numpy(), sample_y.numpy(), ax, title=axtitle, cmap=cmap)


def scatter_density(x, y, ax, title='', cmap='cividis'):
    xy = np.hstack([x, y]).transpose()
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax.scatter(x, y, c=z, s=10, cmap=cmap)
    ax.set_title(title)
    return ax


def compare_ecs_plot(train_marginals, test_marginals, fake_marginals, quantiles, params=None, thresh=None, channel=0, u_x=None, cmap='cividis'):
    """Assumes data provided as marginals unless params are provided"""
    if channel == 1:
        corrs = {'low': (121, 373), 'medium': (294, 189), 'high': (232, 276)}
    elif channel == 0:
        corrs = {'low': (121, 373), 'medium': (294, 189), 'high': (332, 311)}

    fig, axs = plt.subplots(3, 3, figsize=(10, 10), layout='tight')
    train_quantiles = transform_to_quantiles(train_marginals, quantiles, params, thresh)
    test_quantiles = transform_to_quantiles(test_marginals, quantiles, params, thresh)
    fake_quantiles = transform_to_quantiles(fake_marginals, quantiles, params, thresh)

    if u_x is not None:
        h, w, c = u_x.shape
        u_x = u_x.reshape(h * w, c)

    for i, sample_pixels in enumerate([*corrs.values()]):
        ax = axs[i, :]
        plot_sample_density(train_quantiles[..., channel], ax[0], sample_pixels=sample_pixels, cmap=cmap)
        plot_sample_density(test_quantiles[..., channel], ax[1], sample_pixels=sample_pixels, cmap=cmap)
        plot_sample_density(fake_quantiles[..., channel], ax[2], sample_pixels=sample_pixels, cmap=cmap)

        ec = get_ecs(train_marginals[..., channel], sample_pixels)[0]
        ax[0].set_title(f'$\chi$: {ec:.4f}')
        ec = get_ecs(test_marginals[..., channel], sample_pixels)[0]
        ax[1].set_title(f'$\chi$: {ec:.4f}')
        ec = get_ecs(fake_marginals[..., channel], sample_pixels)[0]
        ax[2].set_title(f'$\chi$: {ec:.4f}')

        if u_x is not None:
            u = (u_x[sample_pixels[0], channel], u_x[sample_pixels[1], channel])

            for a in ax:
                a.axhline(u[0], linestyle='dashed', color='k')
                a.axvline(u[1], linestyle='dashed', color='k')

    for ax in axs.ravel():
        ax.set_xlabel(r'wind speed (ms$^{-1}$)')
        ax.set_ylabel(r'wind speed (ms$^{-1}$)')
            
    fig.suptitle(f'Correlations: dimension {channel}')
    return fig


def compare_channels_plot(train_images, test_images, fake_data, cmap='cividis'):
    fig, axs = plt.subplots(3, 3, figsize=(15, 3))

    for i, j in enumerate([300, 201, 102]):

        n, h, w, c = train_images.shape
        data_ravel = tf.reshape(train_images, [n, h * w, c])
        data_sample = tf.gather(data_ravel, j, axis=1).numpy()
        x = np.array([data_sample[:, 0]]).transpose()
        y = np.array([data_sample[:, 1]]).transpose()
        scatter_density(x, y, ax=axs[i, 0], cmap=cmap)

        n, h, w, c = test_images.shape
        data_ravel = tf.reshape(test_images, [n, h * w, c])
        data_sample = tf.gather(data_ravel, j, axis=1).numpy()
        x = np.array([data_sample[:, 0]]).transpose()
        y = np.array([data_sample[:, 1]]).transpose()
        scatter_density(x, y, ax=axs[i, 1], cmap=cmap)

        n, h, w, c = fake_data.shape
        data_ravel = tf.reshape(fake_data, [n, h * w, c])
        data_sample = tf.gather(data_ravel, j, axis=1).numpy()
        x = np.array([data_sample[:, 0]]).transpose()
        y = np.array([data_sample[:, 1]]).transpose()
        scatter_density(x, y, ax=axs[i, 2], cmap=cmap)

        for ax in axs.ravel():
            ax.set_xlabel('u10')
            ax.set_ylabel('v10')
    return fig


def plot_one_hundred_images(fake_data, channel=0, suptitle="Generated marginals", **plot_kwargs):
    fig, axs = plt.subplots(10, 10, layout='tight', figsize=(10, 10))

    for i, ax in enumerate(axs.ravel()):
        im = ax.imshow(fake_data[i, ..., channel], **plot_kwargs)

    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.suptitle(suptitle)

    return fig


