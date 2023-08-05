"""Helper functions for running evtGAN in TensorFlow."""
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import ecdf, genextreme
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

SEED = 42

def tf_unpad(tensor, paddings):
    """Mine: remove Tensor paddings"""
    tensor = tf.convert_to_tensor(tensor)  # incase its a np.array
    unpaddings = [slice(pad.numpy()[0], -pad.numpy()[1]) if sum(pad.numpy()>0)  else slice(None, None) for pad in paddings]
    return tensor[unpaddings]

def frechet_transform(uniform):
    """Apply to Tensor transformed to uniform using ecdf."""
    return - 1 / tf.math.log(uniform)

def gumbel_transform(uniform):
    return -np.log(-np.log(uniform))


def transform_to_marginals(dataset, thresh=.9, fit_tail=False):
    n, h, w, c = dataset.shape
    assert c == 1, "single channel only"
    dataset = dataset[..., 0].reshape(n, h * w)
    marginals, parameters = semiparametric_cdf(dataset, thresh, fit_tail=fit_tail)
    quantiles = interpolate_quantiles(marginals, dataset)
    
    marginals = marginals.reshape(n, h, w, 1)
    parameters = parameters.reshape(h, w, 3)
    quantiles = quantiles.reshape(10000, h, w)
    
    return marginals, parameters, quantiles


def semiparametric_cdf(dataset, thresh, fit_tail=False):
    assert dataset.ndim == 2, "Requires 2 dimensions"
    x = dataset.copy()
    n, J = np.shape(x)
    
    shape = np.empty(J)
    loc = np.empty(J)
    scale = np.empty(J)
    for j in range(J):
        x[:, j], shape[j], loc[j], scale[j] = semiparametric_marginal(x[:, j], thresh=thresh, fit_tail=fit_tail)
    parameters = np.stack([shape, loc, scale], axis=-1)
    return x, parameters


def semiparametric_marginal(x, thresh, fit_tail=False):
    """Heffernan & Tawn (2004). 
    
    Note shape parameter is opposite sign to H&T (2004)."""
    if all(x == 0.):
        return np.array([0.] * len(x)), 0, 0, 0
    
    f = ecdf(x).cdf.evaluate(x)
    
    if fit_tail:
        x_tail = x[f > thresh]
        f_tail = f[f > thresh]
        u_x = max(x[f <= thresh])
        shape, loc, scale = genextreme.fit(x_tail)
        f_tail = 1 - (1 - thresh) * np.maximum(0, (1 - shape * (x_tail - u_x) / scale)) ** (1 / shape)
        f[f >= thresh] = f_tail # distribution no longer uniform
    else:
        f *= 1 - 1e-6
        shape, loc, scale = 0, 0, 0
    return f, shape, loc, scale


def interpolate_quantiles(marginals, quantiles, n=10000):
    """Interpolate the quantiles for inverse transformations."""
    assert quantiles.ndim == 2, "Requires 2 dimensions"
    interpolation_points = np.linspace(0, 1, n)
    J = np.shape(quantiles)[1]
    
    interpolated_quantiles = np.empty((n, J))
    for j in range(J):
        interpolated_quantiles[:, j] = np.interp(interpolation_points, sorted(marginals[..., j]), sorted(quantiles[..., j]))
    return interpolated_quantiles


def transform_to_quantiles(marginals, data, params=None, thresh=None):
    assert marginals.ndim == 4, "Function takes rank 4 arrays"
    n, h, w, c = marginals.shape
    assert data.shape[1:] == (h, w, c), "Marginals and data have different dimensions"
    
    marginals = marginals.reshape(n, h * w, c)
    data = data.reshape(len(data), h * w, c)
    if params is not None:
        params = params.reshape(h * w, 3, c)
    quantiles = []
    for channel in range(c):
        if params is None:
            q = np.array([equantile(marginals[:, j, channel], data[:, j, channel]) for j in range(h * w)]).T
        else:
            q = np.array([equantile(marginals[:, j, channel], data[:, j, channel], params[j, ..., channel], thresh) for j in range(h * w)]).T
        quantiles.append(q)
    quantiles = np.stack(quantiles, axis=-1)
    quantiles = quantiles.reshape(n, h, w, c)
    return quantiles


def equantile(marginals, x, params=None, thresh=None):
    n = len(x)
    x = sorted(x)
    quantiles = np.array([x[int(q * n)] for q in marginals])
    if params is not None:
        u_x = marginals[marginals <= thresh].max()
        marginals_tail = marginals[marginals > thresh]
        quantiles_tail = upper_ppf(marginals_tail, u_x, thresh, params)
        # quantiles_tail = gev_ppf(marginals_tail, params)
        quantiles[marginals > thresh] = quantiles_tail
    return quantiles


def upper_ppf(marginals, u_x, thresh, params):
    """Inverse of (1.3) H&T for $\ksi\leq 0$ and upper tail."""
    shape, scale = params[0], params[2]
    x = u_x + (scale / shape) * (1 - ((1 - marginals) / (1 - thresh))**shape)
    return x


def gev_ppf(q, params):
    """To replace nans with zeros."""
    if sum(params) > 0:
        return genextreme.ppf(q, *params)
    else:
        return 0.

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


def get_ecs(marginals, sample_inds):
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
    # if (uniform == 1).any(): uniform *= 0.9999  # need a TensorFlow analogue
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


def load_marginals_and_quantiles(datadir, train_size=200, datas=['wind_data', 'wave_data', 'precip_data'], paddings=tf.constant([[0,0], [1,1], [1,1], [0,0]])):
    marginals = []
    quantiles = []
    params = []
    for data in datas:
        marginals.append(np.load(os.path.join(datadir, data, 'train', 'marginals.npy'))[..., 0])
        quantiles.append(np.load(os.path.join(datadir, data, 'train', 'quantiles.npy')))
        params.append(np.load(os.path.join(datadir, data, 'train', 'params.npy')))

    marginals = np.stack(marginals, axis=-1)
    quantiles = np.stack(quantiles, axis=-1)
    params = np.stack(params, axis=-1)
    marginals = tf.pad(marginals, paddings)

    # train/valid split
    np.random.seed(2)
    train_inds = np.random.randint(0, marginals.shape[0], train_size)

    marginals_train = np.take(marginals, train_inds, axis=0)
    marginals_test = np.delete(marginals, train_inds, axis=0)
    return marginals_train, marginals_test, quantiles, params

