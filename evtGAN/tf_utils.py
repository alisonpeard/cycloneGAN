"""Helper functions for running evtGAN in TensorFlow."""
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
from scipy.stats import ecdf, genpareto
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


SEED = 42


def tf_unpad(tensor, paddings=tf.constant([[0,0], [1,1], [1,1], [0,0]])):
    """Mine: remove Tensor paddings"""
    tensor = tf.convert_to_tensor(tensor)  # incase its a np.array
    unpaddings = [slice(pad.numpy()[0], -pad.numpy()[1]) if sum(pad.numpy()>0)  else slice(None, None) for pad in paddings]
    return tensor[unpaddings]


def frechet_transform(uniform):
    """Apply to Tensor transformed to uniform using ecdf."""
    return - 1 / tf.math.log(uniform)


def gumbel_transform(uniform):
    return -np.log(-np.log(uniform))


def inverse_gumbel_transform(data):
    uniform = -tf.math.exp(-tf.math.exp(data))
    return uniform


def probability_integral_transform(dataset, distribution="uniform", thresholds=None, fit_tail=False):
    """Transform data to uniform distribution using ecdf."""
    n, h, w, c = dataset.shape
    assert c == 1, "single channel only"
    dataset = dataset[..., 0].reshape(n, h * w)

    if fit_tail is True:
        assert thresholds is not None, "Thresholds must be supplied if fitting tail."
        thresholds = thresholds.reshape(h * w)

    uniform, parameters = semiparametric_cdf(dataset, thresholds, fit_tail=fit_tail)

    uniform = uniform.reshape(n, h, w, 1)
    parameters = parameters.reshape(h, w, 3)

    if distribution == "gumbel":
        transformed = -np.log(-np.log(uniform))
    elif distribution == "uniform":
        transformed = uniform
    else: 
        raise ValueError("Unknown distribution: {distribution}.")
    
    return transformed, parameters


def semiparametric_cdf(dataset, thresh=None, fit_tail=False):
    assert dataset.ndim == 2, "Requires 2 dimensions"
    x = dataset.copy()
    n, J = np.shape(x)

    if not hasattr(thresh, '__len__'):
        thresh = [thresh] * J
    else:
        assert len(thresh) == J, "Thresholds vector must have same length as data."
    
    shape = np.empty(J)
    loc = np.empty(J)
    scale = np.empty(J)
    for j in range(J):
        x[:, j], shape[j], loc[j], scale[j] = semiparametric_marginal_cdf(x[:, j], fit_tail=fit_tail, thresh=thresh[j])
    parameters = np.stack([shape, loc, scale], axis=-1)
    return x, parameters


def semiparametric_marginal_cdf(x, fit_tail=False, thresh=None):
    """Heffernan & Tawn (2004). 
    
    Note shape parameter is opposite sign to Heffernan & Tawn (2004).
    Thresh here is a value, not a percentage."""
    
    if (x.max() - x.min()) == 0.:
        return np.array([0.] * len(x)), 0, 0, 0
    
    f = ecdf(x)  # ecdf(x).cdf.evaluate(x)
    
    if fit_tail:
        assert thresh is not None, "Threshold must be supplied if fitting tail."
        x_tail = x[x > thresh]
        f_tail = f[x > thresh]
        x_tail = x_tail.astype(np.float64)  # otherwise f_thresh gets rounded too low
        shape, loc, scale = genpareto.fit(x_tail, floc=thresh, method="MLE")
        f_thresh = np.interp(thresh, sorted(x), sorted(f)) #ecdf(x).cdf.evaluate(thresh)
        f_tail = 1 - (1 - f_thresh) * (np.maximum(0, (1 - shape * (x_tail - thresh) / scale)) ** (1 / shape))  # second set of parenthesis important
        assert min(f_tail) >= f_thresh, "Error in upper tail calculation."
        f[x > thresh] = f_tail
        f *= 1 - 1e-6
    else:
        shape, loc, scale = 0, 0, 0
        f *= 1 - 1e-6
    return f, shape, loc, scale


def rank(x):
    if x.std() == 0:
        ranked = np.array([len(x) / 2 + 0.5] * len(x))
    else:
        temp = x.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(x))
        ranked = ranks + 1
    return ranked


def ecdf(x):
    return rank(x) / (len(x) + 1)

    
def interpolate_quantiles(marginals, quantiles, n=10000):
    """Interpolate the quantiles for inverse transformations.
    
    No longer using this."""
    assert quantiles.ndim == 2, "Requires 2 dimensions"
    interpolation_points = np.linspace(0, 1, n)
    J = np.shape(quantiles)[1]
    
    interpolated_quantiles = np.empty((n, J))
    for j in range(J):
        interpolated_quantiles[:, j] = np.interp(interpolation_points, sorted(marginals[..., j]), sorted(quantiles[..., j]))
    return interpolated_quantiles


def inv_probability_integral_transform(marginals, x, y, params=None, thresh=None):
    """Transform uniform marginals to original distributions, by inverse-interpolating ecdf."""
    assert marginals.ndim == 4, "Function takes rank 4 arrays"
    n, h, w, c = marginals.shape
    assert x.shape[1:] == (h, w, c), "Marginals and x have different dimensions."
    assert y.shape[1:] == (h, w, c), "Marginals and y have different dimensions."
    assert x.shape[0] == y.shape[0], "x and y have different dimensions."
    
    marginals = marginals.reshape(n, h * w, c)
    x = x.reshape(len(x), h * w, c)
    y = y.reshape(len(y), h * w, c)

    if params is not None:
        assert params.shape == (h, w, 3, c), "Marginals and parameters have different dimensions."
        params = params.reshape(h * w, 3, c)
    
    quantiles = []
    for channel in range(c):
        if params is None:
            q = np.array([empirical_quantile(marginals[:, j, channel], x[:, j, channel], y[:, j, channel]) for j in range(h * w)]).T
        else:
            if hasattr(thresh, "__len__"):
                thresh = thresh.reshape(h * w, c)
            else:
                thresh = [thresh] * (h * w, c)
            q = np.array([empirical_quantile(marginals[:, j, channel], x[:, j, channel], y[:, j, channel], params[j, ..., channel], thresh[j, channel]) for j in range(h * w)]).T
        quantiles.append(q)
    
    quantiles = np.stack(quantiles, axis=-1)
    quantiles = quantiles.reshape(n, h, w, c)
    return quantiles


def empirical_quantile(marginals, x, y, params=None, thresh=None):
    """(Semi)empirical quantile/percent/point function.
    
    x [was] a vector of interpolated quantiles of data (usually 100,000)
    Now x and y are data that original marginals were calculated from, where x
    is data and y corresponding densities."""
    n = len(x)
    x = sorted(x)

    if (marginals.max() - marginals.min()) == 0.: # all identical pixels
        return np.array([-999] * len(marginals))

    if marginals.max() >= 1:
        warnings.warn("Some marginals >= 1.")
        marginals *= 1 - 1e-6
    quantiles = np.interp(marginals, sorted(y), sorted(x))
    if params is not None:
        f_thresh = np.interp(thresh, sorted(x), sorted(y))
        if len(quantiles[marginals <= f_thresh]) > 0:
            u_x = quantiles[marginals <= f_thresh].max()
        else:  # just a catch for mad marginals, but shouldn't happen
            warnings.warn("All marginals above u_x.")
            u_x = quantiles.min()
        marginals_tail = marginals[marginals > f_thresh]
        quantiles_tail = upper_ppf(marginals_tail, u_x, f_thresh, params)
        quantiles[marginals > f_thresh] = quantiles_tail
    return quantiles


def upper_ppf(marginals, u_x, thresh, params):
    """Inverse of (1.3) H&T for $\ksi\leq 0$ and upper tail."""
    shape, scale = params[0], params[2]
    x = u_x + (scale / shape) * (1 - ((1 - marginals) / (1 - thresh))**shape)
    return x


# def gev_ppf(q, params):
#     """To replace nans with zeros."""
#     if sum(params) > 0:
#         return genpareto.ppf(q, *params)
#     else:
#         return 0.


def get_ecs(marginals, sample_inds):
    data = tf.cast(marginals, dtype=tf.float32)
    n, h, w = tf.shape(data)[:3]
    data = tf.reshape(data, [n, h * w])
    data = tf.gather(data, sample_inds, axis=1)
    frechet = tf_inv_frechet(data)
    ecs = []
    for i in range(len(sample_inds)):
        for j in range(i):
            ecs.append(raw_extremal_correlation(frechet[:, i], frechet[:, j]))
    return ecs


def tf_exp(uniform):
    # if (uniform == 1).any(): uniform *= 0.9999  # need a TensorFlow analogue
    exp_distributed = -tf.math.log(1 - uniform)
    return exp_distributed


def tf_inv_frechet(uniform):
    # if (uniform == 1).any(): uniform *= 0.9999  # need a TensorFlow analogue
    exp_distributed = -tf.math.log(uniform)
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
    images = []
    params = []
    thresholds = []
    threshold_ecdfs = []
    for data in datas:
        marginals.append(np.load(os.path.join(datadir, data, 'train', 'marginals.npy'))[..., 0])
        quantiles.append(np.load(os.path.join(datadir, data, 'train', 'quantiles.npy')))
        params.append(np.load(os.path.join(datadir, data, 'train', 'params.npy')))
        images.append(np.load(os.path.join(datadir, data, 'train', 'images.npy'))[..., 0])
        thresholds.append(np.load(os.path.join(datadir, data, 'train', 'thresholds.npy')))
        threshold_ecdfs.append(np.load(os.path.join(datadir, data, 'train', 'threshold_ecdfs.npy')))


    marginals = np.stack(marginals, axis=-1)
    quantiles = np.stack(quantiles, axis=-1)
    params = np.stack(params, axis=-1)
    images = np.stack(images, axis=-1)
    thresholds = np.stack(thresholds, axis=-1)
    threshold_ecdfs = np.stack(threshold_ecdfs, axis=-1)

    # paddings
    marginals = tf.pad(marginals, paddings)

    # train/valid split
    np.random.seed(2)
    #train_inds = np.random.randint(0, marginals.shape[0], train_size)
    train_inds = np.random.choice(np.arange(0, marginals.shape[0], 1), size=train_size, replace=False)

    marginals_train = np.take(marginals, train_inds, axis=0)
    marginals_test = np.delete(marginals, train_inds, axis=0)
    return marginals_train, marginals_test, quantiles, params, images, thresholds, threshold_ecdfs

