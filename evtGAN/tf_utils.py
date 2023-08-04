"""Helper functions for running evtGAN in TensorFlow."""
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import genextreme
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.stats import gaussian_kde
from scipy import stats

SEED = 42

def tf_unpad(tensor, paddings):
    """Mine: remove Tensor paddings"""
    unpaddings = [slice(pad.numpy()[0], -pad.numpy()[1]) if sum(pad.numpy()>0)  else slice(None, None) for pad in paddings]
    return tensor[unpaddings]

def frechet_transform(uniform):
    """Apply to Tensor transformed to uniform using ecdf."""
    return - 1 / tf.math.log(uniform)

def transform_to_marginals(dataset):
    assert dataset.ndim == 4, "Function takes rank 4 arrays."
    n, h, w, c = dataset.shape
    dataset = dataset.reshape(n, h * w, c)
    marginals = []
    for channel in range(c):
        marginals.append(marginal(dataset[..., channel]))
    marginals = np.stack(marginals, axis=-1)
    marginals = marginals.reshape(n, h, w, c)
    return marginals
        

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


def transform_to_quantiles(marginals, data):
    assert marginals.ndim == 4, "Function takes rank 4 arrays"
    n, h, w, c = marginals.shape
    assert data.shape[1:] == (h, w, c), "Marginals and data have different dimensions"
    
    marginals = marginals.reshape(n, h * w, c)
    data = data.reshape(len(data), h * w, c)
    quantiles = []
    for channel in range(c):
        q = np.array([equantile(marginals[:, j, channel], data[:, j, channel]) for j in range(h * w)]).T
        quantiles.append(q)
    quantiles = np.stack(quantiles, axis=-1)
    quantiles = quantiles.reshape(n, h, w, c)
    return quantiles


def equantile(quantiles, x):
    n = len(x)
    x = sorted(x)
    return [x[int(q * n)] for q in quantiles]

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


def get_ecs(marginals, sample_inds, params=None):
    """TODO: If data is not already transformed to marginals, provide params"""
    # process to nice shape
    data = tf.cast(marginals, dtype=tf.float32)
    n, h, w = tf.shape(data)[:3]
    data = tf.reshape(data, [n, h * w])
    data = tf.gather(data, sample_inds, axis=1)

    frechet = tf_inv_exp(data)

    ecs = []
    for i in range(len(sample_inds)):
        for j in range(i):
            ecs.append(raw_extremal_correlation(frechet[:, i], frechet[:, j]))
    return ecs


def tf_inv_exp(uniform):
    if (uniform == 1).any(): uniform *= 0.9999
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
        tf.print("Warning: all zeros in minima array.")
        theta = 2
    return theta


def gaussian_blur(img, kernel_size=11, sigma=5):
    """See: https://gist.github.com/blzq/c87d42f45a8c5a53f5b393e27b1f5319"""
    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],
                                  padding='SAME', data_format='NHWC')


########################################################################################################

def load_data(datadir, imsize=(18, 22), conditions='all', dim=None):
    """Load wind image data to correct size."""
    assert conditions in ['cyclone', 'all'], "Invalid conditions."
    datas = []
    df = pd.read_csv(os.path.join(datadir, f"{dim}_dailymax.csv"), index_col=[0])
    if conditions == "cyclone":
        df = df[df['cyclone_flag']]
    cyclone_flag = df['cyclone_flag'].values
    df = df.drop(columns=['time', 'cyclone_flag'])
    values = df.values.astype(float)
    ngrids = int(np.sqrt(values.shape[1]))
    values = values.reshape(values.shape[0], ngrids, ngrids)
    values = np.flip(values, axis=[1])
    values = values[..., np.newaxis]
    data = tf.image.resize(values, (imsize[0], imsize[1]))
    return data.numpy(), cyclone_flag


def load_marginals(datadir, train_size=200, imsize=(18, 22), conditions="all", shuffle=False, paddings=tf.constant([[0,0], [1,1], [1,1], [0,0]])):
    """Load wind data, split into train and test and converted to marginals based-on the train set."""
    assert conditions in ['cyclone', 'all'], "Invalid conditions."
    h, w = imsize
    winds, cyclone_flag = load_winds(datadir, imsize, conditions)
    n = winds.shape[0]
    assert n > train_size, "Train size ({train_size}) > dataset size ({n})."
    
    if shuffle: 
        index = [*range(n)]
        np.random.seed(42)  # will always create same dataset
        np.random.shuffle(index)
        winds = winds[index]
        cyclone_flag = cyclone_flag[index]
    
    train, test = winds[:train_size, ...], winds[train_size:, ...]
    quantiles = train.copy()
    train = transform_to_marginals(train)
    test = transform_to_marginals(test)
    train = tf.pad(train, paddings)
    test = tf.pad(test, paddings)
    return train, test, quantiles, cyclone_flag


def cyclone_transfer_set(train, test, cyclone_flag, train_size):
    """Create a cyclone-only training set for transfer learning."""
    ncyclones = sum(cyclone_flag) // 2
    cyclone_train = train[cyclone_flag[:train_size]]
    cyclone_test = test[cyclone_flag[train_size:]]

    more_train = cyclone_test[:ncyclones - cyclone_train.shape[0]]
    cyclone_test = cyclone_test[ncyclones - cyclone_train.shape[0]:]
    cyclone_train = tf.concat([cyclone_train, more_train], axis=0)
    return cyclone_train, cyclone_test