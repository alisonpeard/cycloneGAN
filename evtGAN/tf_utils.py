"""Helper functions for running evtGAN in TensorFlow."""

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.stats import gaussian_kde
from scipy import stats

SEED = 42

def kelvinToCelsius(kelvin):
    return (kelvin - 273.15)


def tf_unpad(tensor, paddings):
    """Mine: remove Tensor paddings"""
    unpaddings = [slice(pad.numpy()[0], -pad.numpy()[1]) if sum(pad.numpy()>0)  else slice(None, None) for pad in paddings]
    return tensor[unpaddings]


def marginal(dataset):
    a = ecdf(dataset)
    J = np.shape(a)[1]
    n = np.shape(a)[0]
    z = n * (-1)
    for j in range(J):
        if np.sum(a[:, j]) == z:
            a[:, j] = np.zeros(np.shape(a[:, j])[0])
    return a


def ecdf(dataset):
    return rank(dataset) / (len(dataset) + 1)
# NOTE: (tf_utils.ecdf(train_set) == tf_utils.marginal(train_set)).all() is true usually


def rank(dataset):
    ranked = np.empty(np.shape(dataset))
    for j in range(np.shape(ranked)[1]):
        if all(i == dataset[0,j] for i in dataset[:,j]):
            ranked[:,j] = len(ranked[:,j]) / 2 + 0.5
        else:
            array = dataset[:,j]
            temp = array.argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(array))
            ranked[:,j] = ranks + 1
    return ranked


def get_ec_variogram(x, y):
    """Extremal correlation by variogram as in §2.1."""
    variogram_xy = np.mean((x - y)**2)
    ec_xy = 2 - 2 * stats.norm.cdf(np.sqrt(variogram_xy) / 2)
    return ec_xy


def frechet_transform(uniform):
    """Apply to Tensor transformed to uniform using ecdf."""
    return - 1 / tf.math.log(uniform)



####################################################
# Tail dependence (Χ) calculations
####################################################
def tail_dependence_diff(data, noise, sample_size):
    """Get mean l2-norm of tail dependence metric."""
    n, h, w, _ = tf.shape(data)
    sample_inds = tf.random.uniform([25], maxval=(h * w), dtype=tf.dtypes.int32)
    generated_data = generator(noise)

    ecs_tr = get_ecs(data, sample_inds)
    ecs_gen = get_ecs(generated_data, sample_inds)

    l2_tr = tf.math.sqrt(tf.reduce_mean((tf.stack(ecs_gen) - tf.stack(ecs_tr))**2))
    return l2_tr

def get_ecs(data, sample_inds):
    # process to nice shape
    n, h, w, _ = tf.shape(data)
    data = tf.reshape(data, [n, h * w])
    data = tf.gather(data, sample_inds, axis=1)

    # transform to uniform distribuion over each marginal then Fréchet (not fully justified)
    uniform = tf.map_fn(lambda x: tfd.Empirical(data[:, x]).cdf(data[:, x])*(n/(n+1)), tf.range(tf.shape(data)[1]), dtype=tf.float32)
    frechet = tf_inv_exp(uniform)

    ecs = []
    for i in range(len(sample_inds)):
        for j in range(i):
            ecs.append(raw_extremal_correlation(frechet[i], frechet[j]))

    return ecs

def tf_inv_exp(uniform):
    uniform *= 0.9999
    exp_distributed = -tf.math.log(1-uniform)
    return exp_distributed


def raw_extremal_correlation(frechet_x, frechet_y):
    """Where x and y have been transformed to their Fréchet marginal distributions.

    ..[1] Max-stable process and spatial extremes, Smith (1990)
    """
    n = tf.shape(frechet_x)[0]
    assert n == tf.shape(frechet_y)[0]
    n = tf.cast(n, dtype=tf.float32)
    minima = tf.reduce_sum(tf.math.minimum(frechet_x, frechet_y))
    if tf.greater(tf.reduce_sum(minima), 0):
        theta = n / minima
    else:
        tf.print("All zeros in minima array.")
        theta = 2
    return theta

########################################################################################################


def plot_sample_density(data, reference_data, ax, celcius=True, sample_pixels=None):
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

    frechet_x = -tf.math.log(1 - sample_x)
    frechet_y = -tf.math.log(1 - sample_y)

    ec_xy = raw_extremal_correlation(frechet_x, frechet_y)

    # transform to original data and Celcius and plot
    sample_x = np.quantile(reference_data, sample_x)
    sample_y = np.quantile(reference_data, sample_y)

    if celcius:
        sample_x = kelvinToCelsius(sample_x)
        sample_y = kelvinToCelsius(sample_y)

    scatter_density(sample_x, sample_y, ax, title=f'$\chi$: {ec_xy:4f}')


def scatter_density(x, y, ax, title=''):
    xy = np.hstack([x, y]).transpose()
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax.scatter(x, y, c=z, s=10)
    ax.set_title(title)
    return ax


def load_for_tf(root, paddings, n_train, output_size=None):
    df_train_test = pd.read_csv(root, sep=',',header=None).iloc[1:]

    #Load data and view
    plt.figure(figsize=(15, 3))
    plt.plot(range(2000), df_train_test.iloc[0,:].values.astype(float), linewidth=.5, color='k')
    plt.ylabel('Temperature (K)')
    plt.title('Time series for single pixel')
    plt.show()

    df_train_test = df_train_test.values.astype(float).transpose()
    n = df_train_test.shape[0]

    #train set: need to standardise it separately (as df is ... but then split)
    train_set = df_train_test[:n_train,:]

    #valid set: need to standardise it separately (as df is ... but then split)
    test_set = df_train_test[n_train:,:]

    train = tf.convert_to_tensor(train_set, dtype='float32')
    train = tf.reshape(train, (n_train, 18, 22, 1))
    if output_size is not None:
        train = tf.image.resize(train, [output_size[0]-1, output_size[1]-1])
#     train = tf.pad(train, paddings)

    test = tf.convert_to_tensor(test_set, dtype='float32')
    test = tf.reshape(test, (n - n_train, 18, 22, 1))
    if output_size is not None:
        test = tf.image.resize(test, [output_size[0]-1, output_size[1]-1])

    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axs):
        im = ax.imshow(train[i, :, :, 0].numpy())
    plt.suptitle('Training data (original scale)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    return train, test


def load_and_scale_for_tf(root, paddings, n_train, output_size=None):
    df_train_test = pd.read_csv(root, sep=',',header=None).iloc[1:]

    #Load data and view
    plt.figure(figsize=(15, 3))
    plt.plot(range(2000), df_train_test.iloc[0, :].values.astype(float), linewidth=.5, color='k')
    plt.ylabel('Temperature (K)')
    plt.title('Time series for single pixel')
    plt.show()

    df_train_test = df_train_test.values.astype(float).transpose()
    n = df_train_test.shape[0]

    #train set: need to standardise it separately (as df is ... but then split)
    train_set = df_train_test[:n_train,:]
    train_set = marginal(train_set)

    #valid set: need to standardise it separately (as df is ... but then split)
    test_set = df_train_test[n_train:,:]
    test_set = marginal(test_set)

    train = tf.convert_to_tensor(train_set, dtype='float32')
    train = tf.reshape(train, (n_train, 18, 22, 1))
    if output_size is not None:
        train = tf.image.resize(train, [output_size[0]-1, output_size[1]-1])
    train = tf.pad(train, paddings)

    test = tf.convert_to_tensor(test_set, dtype='float32')
    test = tf.reshape(test, (n - n_train, 18, 22, 1))
    if output_size is not None:
        test = tf.image.resize(test, [output_size[0]-1, output_size[1]-1])
    test = tf.pad(test, paddings)

    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axs):
        im = ax.imshow(train[i, :, :, 0].numpy())
    plt.suptitle('Training data (scaled)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    return train, test


def load_era5_for_tf(indir, n_train, output_size=(17, 21), conditions='normal', viz=True):
    """Load ERA5 data from supplied directories.

    Parameters:
    -----------
    indir : string
    n_train : int
    output_size : tuple of ints, default=(17, 21)
    conditions : string, default='normal'
    viz : bool, default=True
    """
    df = pd.read_csv(indir)
    if conditions == 'normal':
        df = df[~df['cyclone_flag']]
    elif conditions == 'cyclone':
        df = df[df['cyclone_flag']]
    elif conditions == 'all':
        pass
    else:
        raise Exception("condition must be one of ['normal', 'cyclone', 'all']")
    df = df.drop(columns=['cyclone_flag', 'time'])

    n = df.shape[1]

    #Load data and view
    if viz:
        plt.figure(figsize=(15, 3))
        plt.plot(range(n), df.iloc[0, :].values.astype(float), linewidth=.5, color='k')
        plt.ylabel('Wind u10 (mps)')
        plt.title('Time series for single pixel')
        plt.show()

    df = df.values.astype(float)

    n = df.shape[0]
    print(f"Dataset has {n} entries.")
    assert n_train < n, f"Requested train size ({n_train}) exceeds size of dataset ({n})."

    train_set = df[:n_train,:]
    test_set = df[n_train:,:]

    train = tf.convert_to_tensor(train_set, dtype='float32')
    train = tf.reshape(train, (n_train, 61, 61, 1))

    if output_size is not None:
        train = tf.image.resize(train, [output_size[0]-1, output_size[1]-1])

    test = tf.convert_to_tensor(test_set, dtype='float32')
    test = tf.reshape(test, (n - n_train, 61, 61, 1))
    if output_size is not None:
        test = tf.image.resize(test, [output_size[0]-1, output_size[1]-1])

    if viz:
        fig, axs = plt.subplots(1, 5, figsize=(15, 3))
        for i, ax in enumerate(axs):
            im = ax.imshow(train[i, :, :, 0].numpy())
        plt.suptitle('Training data (original scale)')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

    return train, test



def load_and_scale_era5_for_tf(indir, paddings, n_train, output_size = (17, 21), conditions='normal', viz=False):

    df = pd.read_csv(indir)
    if conditions == 'normal':
        df = df[~df['cyclone_flag']]
    elif conditions == 'cyclone':
        df = df[df['cyclone_flag']]
    elif conditions == 'all':
        pass
    else:
        raise Exception("condition must be one of ['normal', 'cyclone', 'all']")

    df = df.drop(columns=['cyclone_flag', 'time'])
    n = df.shape[1]

    #Load data and view
    if viz:
        plt.figure(figsize=(15, 3))
        plt.plot(range(n), df.iloc[0, :].values.astype(float), linewidth=.5, color='k')
        plt.ylabel('Wind u10 (mps)')
        plt.title('Time series for single pixel')
        plt.show()

    df = df.values.astype(float)

    n = df.shape[0]
    print(f"Dataset has {n} entries.")
    assert n_train < n, f"Requested train size ({n_train}) exceeds size of dataset ({n})."

    train_set = df[:n_train,:]
    train_set = marginal(train_set)
    test_set = df[n_train:,:]
    test_set = marginal(test_set)

    train = tf.convert_to_tensor(train_set, dtype='float32')
    train = tf.reshape(train, (n_train, 61, 61, 1))

    if output_size is not None:
        train = tf.image.resize(train, [output_size[0]-1, output_size[1]-1])
    train = tf.pad(train, paddings)

    test = tf.convert_to_tensor(test_set, dtype='float32')
    test = tf.reshape(test, (n - n_train, 61, 61, 1))
    if output_size is not None:
        test = tf.image.resize(test, [output_size[0]-1, output_size[1]-1])
    test = tf.pad(test, paddings)

    if viz:
        fig, axs = plt.subplots(1, 5, figsize=(15, 3))
        for i, ax in enumerate(axs):
            im = ax.imshow(train[i, :, :, 0].numpy())
        plt.suptitle('Training data (original scale)')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

    return train, test


def load_era5_datasets(roots, n_train, batch_size, im_size, paddings=tf.constant([[0,0],[1,1],[1,1], [0,0]]), conditions="normal", scale=True, viz=False):
    """Wrapper function to load ERA5 with certain conditions (for hyperparameter tuning)."""
    train_sets = []
    test_sets = []

    for root in roots:
        if scale:
            train_images, test_images = load_and_scale_era5_for_tf(root, paddings, n_train, im_size, conditions=conditions, viz=viz)
        else:
            train_images, test_images = load_era5_for_tf(root, n_train, im_size, conditions=conditions, viz=viz)
        train_sets.append(train_images[..., 0])
        test_sets.append(test_images[..., 0])

    # stack them together in multichannel images
    train_images = tf.stack(train_sets, axis=-1)
    test_images = tf.stack(test_sets, axis=-1)

    # create datasets
    train = tf.data.Dataset.from_tensor_slices(train_images).shuffle(len(train_images)).batch(batch_size)
    test = tf.data.Dataset.from_tensor_slices(test_images).shuffle(len(test_images)).batch(batch_size)

    return train, test, train_images, test_images
