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

def kelvinToCelsius(kelvin):
    return (kelvin - 273.15)


def tf_unpad(tensor, paddings):
    """Mine: remove Tensor paddings"""
    unpaddings = [slice(pad.numpy()[0], -pad.numpy()[1]) if sum(pad.numpy()>0)  else slice(None, None) for pad in paddings]
    return tensor[unpaddings]

def fit_gev(dataset):
    """Fit GEV to dataset. Takes a while for large datasets."""
    m = dataset.shape[1]
    params = [genextreme.fit(dataset[..., j]) for j in range(m)]
    return params

def gev_marginals(dataset, params):
    """Transform dataset to marginals of generalised extreme value distribution."""
    m = dataset.shape[1]
    marginals = np.array([genextreme.cdf(dataset[..., j], *params[j]) for j in range(m)]).T
    return marginals

def gev_quantiles(marginals, params):
    """Transform marginals of generalised extreme value distribution to original scale."""
    m = marginals.shape[1]
    quantiles = np.array([genextreme.ppf(marginals[..., j], *params[j]) for j in range(m)]).T
    return quantiles

# marginals_to_winds
def marginals_to_winds(marginals, params:tuple):
    winds = tf.image.resize(marginals, [61, 61])
    winds = tf.reshape(winds, [1000, 61 * 61, 2]).numpy()
    winds_u10 = gev_quantiles(winds[..., 0], params[0])
    winds_u10 = np.reshape(winds_u10, [1000, 61, 61])
    winds_v10 = gev_quantiles(winds[..., 0], params[1])
    winds_v10 = np.reshape(winds_v10, [1000, 61, 61])
    winds = np.stack([winds_u10, winds_v10], axis=-1)
    return winds

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

def load_marginals(indir, conditions="all", dim="u10", output_size=(19, 23), paddings=tf.constant([[0,0],[1,1],[1,1],[0,0]])):
    cyclone_marginals_train = np.load(os.path.join(indir, f"cyclone_marginals_{dim}_train.npy"))
    normal_marginals_train = np.load(os.path.join(indir, f"normal_marginals_{dim}_train.npy"))
    cyclone_marginals_test = np.load(os.path.join(indir, f"cyclone_marginals_{dim}_test.npy"))
    normal_marginals_test = np.load(os.path.join(indir, f"normal_marginals_{dim}_test.npy"))

    if conditions == "all":
        train = np.vstack([cyclone_marginals_train, normal_marginals_train])
        test = np.vstack([cyclone_marginals_test, normal_marginals_test])
    elif conditions == "normal":
        train = normal_marginals_train
        test = normal_marginals_test
    elif conditions == "cyclone":
        train = cyclone_marginals_train
        test = cyclone_marginals_test

    train_size = len(train)
    train = tf.convert_to_tensor(train, dtype='float32')
    train = tf.reshape(train, (train_size, 61, 61, 1))
    train = tf.image.resize(train, [output_size[0]-1, output_size[1]-1])
    train = tf.pad(train, paddings)

    test_size = len(test)
    test = tf.convert_to_tensor(test, dtype='float32')
    test = tf.reshape(test, (test_size, 61, 61, 1))
    test = tf.image.resize(test, [output_size[0]-1, output_size[1]-1])
    test = tf.pad(test, paddings)

    return train, test

def load_winds(indir, conditions="all", dim="u10"):
    cyclone_train = np.load(os.path.join(indir, f"cyclone_original_{dim}_train.npy"))
    normal_train = np.load(os.path.join(indir, f"normal_original_{dim}_train.npy"))
    cyclone_test = np.load(os.path.join(indir, f"cyclone_original_{dim}_test.npy"))
    normal_test = np.load(os.path.join(indir, f"normal_original_{dim}_test.npy"))

    if conditions == "all":
        train = np.vstack([cyclone_train, normal_train])
        test = np.vstack([cyclone_test, normal_test])
    elif conditions == "normal":
        train = normal_train
        test = normal_test
    elif conditions == "cyclone":
        train = cyclone_train
        test = cyclone_test

    train_size = len(train)
    train = tf.convert_to_tensor(train, dtype='float32')
    train = tf.reshape(train, (train_size, 61, 61, 1))

    test_size = len(test)
    test = tf.convert_to_tensor(test, dtype='float32')
    test = tf.reshape(test, (test_size, 61, 61, 1))
    return train, test


def load_test_images(indir, train_size, conditions="all", dims=["u10", "v10"]):
    """Wrapper function to load ERA5 with certain conditions (for hyperparameter tuning)."""
    indir = os.path.join(indir, f"train_{train_size}")
    test_sets = []
    train_sets = []
    for dim in dims:
        train, test = load_winds(indir, conditions, dim)
        train_sets.append(train[..., 0])
        test_sets.append(test[..., 0])
    train_ims = tf.stack(train_sets, axis=-1)
    test_ims = tf.stack(test_sets, axis=-1)
    return test_ims


def load_datasets(indir, train_size, batch_size, conditions="all", dims=["u10", "v10"], output_size=(19, 23), paddings=tf.constant([[0,0],[1,1],[1,1],[0,0]])):
    """Wrapper function to load ERA5 with certain conditions (for hyperparameter tuning)."""

    indir = os.path.join(indir, f"train_{train_size}")
    train_sets = []
    test_sets = []
    for dim in dims:
        train, test = load_marginals(indir, dim=dim, conditions=conditions, output_size=output_size, paddings=paddings)
        train_sets.append(train[..., 0])
        test_sets.append(test[..., 0])

    # stack them together in multichannel images
    train_ims = tf.stack(train_sets, axis=-1)
    test_ims = tf.stack(test_sets, axis=-1)

    # create datasets
    train = tf.data.Dataset.from_tensor_slices(train_ims).shuffle(len(train_ims)).batch(batch_size)
    test = tf.data.Dataset.from_tensor_slices(test_ims).shuffle(len(test_ims)).batch(batch_size)

    return train, test
