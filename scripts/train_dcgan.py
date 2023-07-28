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
paddings = tf.constant([[0,0], [1,1], [1,1], [0,0]])
var = 'wind'
conditions = "all"
im_size = (19, 23)
cwd = os.getcwd()
wd = os.path.join(cwd, "..")
indir = "/Users/alison/Documents/DPhil/multivariate/processed_wind_data"
imdir = os.path.join(wd, 'figures', 'temp')


def log_image_to_wandb(fig, name:str, dir:str):
    impath = os.path.join(dir, f"{name}.png")
    fig.savefig(impath, **plot_kwargs)
    wandb.log({name: wandb.Image(impath)})


def main(config):
    # load data
    train, test = tf_utils.load_datasets(indir, config.train_size, config.batch_size, conditions=conditions)
    train_images, test_images = tf_utils.load_test_images(indir, config.train_size, conditions=conditions)
    train_images = train_images.numpy()
    test_images = test_images.numpy()

    params_u10 = np.load(os.path.join(indir, f"train_{config.train_size}", "gev_params_u10_train.npy"))
    params_v10 = np.load(os.path.join(indir, f"train_{config.train_size}", "gev_params_v10_train.npy"))
    params_u10_test = np.load(os.path.join(indir, f"train_{config.train_size}", "gev_params_u10_test.npy"))
    params_v10_test = np.load(os.path.join(indir, f"train_{config.train_size}", "gev_params_v10_test.npy"))

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
    fake_winds = tf_utils.marginals_to_winds(fake_marginals, (params_u10, params_v10))
    fake_marginals = fake_marginals.numpy()

    fig = viz_utils.plot_generated_marginals(fake_marginals)
    log_image_to_wandb(fig, f'generated_marginals', imdir)

    fig = viz_utils.compare_ecs_plot(train_images, test_images, fake_winds, params_u10, params_u10_test, channel=0)
    log_image_to_wandb(fig, 'correlations_u10', imdir)

    fig = viz_utils.compare_ecs_plot(train_images, test_images, fake_winds, params_u10, params_u10_test, channel=1)
    log_image_to_wandb(fig, 'correlations_v10', imdir)

    fig = viz_utils.compare_channels_plot(train_images, test_images, fake_winds)
    log_image_to_wandb(fig, 'correlations multivariate', imdir)
    plt.show()


if __name__ == "__main__":
    wandb.init(settings=wandb.Settings(code_dir="."))  # saves snapshot of code as artifact (less useful now)

    rundir = os.path.join(cwd, "saved-models", wandb.run.name)
    os.makedirs(rundir)

    tf.keras.utils.set_random_seed(wandb.config['seed'])  # sets seeds for base-python, numpy and tf
    tf.config.experimental.enable_op_determinism()  # removes stochasticity from individual operations
    main(wandb.config)
