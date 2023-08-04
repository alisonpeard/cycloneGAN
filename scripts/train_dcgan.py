"""
Note, requires config to create new model too.
>>> new_gan = DCGAN.DCGAN(config)
>>> new_gan.generator.load_weights(os.path.join(wd, 'saved_models', f'{finish_time}_generator_weights'))
>>> new_gan.discriminator.load_weights(os.path.join(wd, 'saved_models', f'{finish_time}_discriminator_weights'))
"""

import os
import numpy as np
from datetime import datetime
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import wandb
from wandb.keras import WandbCallback

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from evtGAN import ChiScore, CrossEntropy, DCGAN, tf_utils, viz_utils, compile_dcgan

global rundir

plot_kwargs = {'bbox_inches': 'tight', 'dpi': 300}

# some static variables
cwd = os.getcwd()
wd = os.path.join(cwd, "..")
datadir = "/Users/alison/Documents/DPhil/multivariate/wind_data"
imdir = os.path.join(wd, 'figures', 'temp')
paddings=tf.constant([[0,0], [1,1], [1,1], [0,0]])


def log_image_to_wandb(fig, name:str, dir:str):
    impath = os.path.join(dir, f"{name}.png")
    fig.savefig(impath, **plot_kwargs)
    wandb.log({name: wandb.Image(impath)})


def main(config):
    # load data
    train_marginals, test_marginals, quantiles, cyclone_flag = tf_utils.load_marginals(datadir, config.train_size, shuffle=True, paddings=paddings)
    train = tf.data.Dataset.from_tensor_slices(train_marginals).batch(config.batch_size)
    test = tf.data.Dataset.from_tensor_slices(test_marginals).batch(config.batch_size)

    # train test callbacks
    chi_score = ChiScore({'train': next(iter(train)), 'test': next(iter(test))}, frequency=config.chi_frequency)
    cross_entropy = CrossEntropy(next(iter(test)))

    # compile
    with tf.device('/gpu:0'):
        gan = compile_dcgan(config)
        gan.generator.load_weights("/Users/alison/Documents/DPhil/multivariate/saved_models/lilac-pine-10_generator_weights")
        gan.discriminator.load_weights("/Users/alison/Documents/DPhil/multivariate/saved_models/lilac-pine-10_discriminator_weights")
        gan.fit(train, epochs=config.nepochs, callbacks=[WandbCallback(), chi_score, cross_entropy])

    gan.generator.save_weights(os.path.join(rundir, f'generator_weights'))
    gan.discriminator.save_weights(os.path.join(rundir, f'discriminator_weights'))

    # generate 1000 images to visualise some results
    fake_marginals = gan(1000)
    fake_marginals = tf_utils.tf_unpad(fake_marginals, paddings)
    # fake_quantiles = tf_utils.transform_to_quantiles(fake_marginals, quantiles)

    fig = viz_utils.plot_generated_marginals(fake_marginals)
    log_image_to_wandb(fig, f'generated_marginals', imdir)

    fig = viz_utils.compare_ecs_plot(train_marginals, test_marginals, fake_marginals, quantiles, channel=0)
    log_image_to_wandb(fig, 'correlations_u10', imdir)

    fig = viz_utils.compare_ecs_plot(train_marginals, test_marginals, fake_marginals, quantiles, channel=1)
    log_image_to_wandb(fig, 'correlations_v10', imdir)

    fig = viz_utils.compare_channels_plot(train_marginals, test_marginals, fake_marginals)
    log_image_to_wandb(fig, 'correlations multivariate', imdir)
    plt.show()


if __name__ == "__main__":
    wandb.init(settings=wandb.Settings(code_dir="."))  # saves snapshot of code as artifact (less useful now)

    rundir = os.path.join(wd, "saved-models", wandb.run.name)
    os.makedirs(rundir)

    tf.keras.utils.set_random_seed(wandb.config['seed'])  # sets seeds for base-python, numpy and tf
    tf.config.experimental.enable_op_determinism()  # removes stochasticity from individual operations
    main(wandb.config)
