"""
Note, requires config to create new model too.
>>> new_gan = DCGAN.DCGAN(config)
>>> new_gan.generator.load_weights(os.path.join(wd, 'saved_models', f'{finish_time}_generator_weights'))
>>> new_gan.discriminator.load_weights(os.path.join(wd, 'saved_models', f'{finish_time}_discriminator_weights'))
"""

import os
from datetime import datetime
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import wandb
from wandb.keras import WandbCallback

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from evtGAN import DCGAN, tf_utils, viz_utils

global rundir

plot_kwargs = {'bbox_inches': 'tight', 'dpi': 300}

# some static variables
paddings = tf.constant([[0,0],[1,1],[1,1], [0,0]])
var = 'wind'
conditions = "all"
im_size = (19, 23)
cwd = os.getcwd()
wd = os.path.join(cwd, "..")
roots = [os.path.join(wd, "..", "wind_data", "u10_dailymax.csv"), os.path.join(wd, "..", "wind_data", "v10_dailymax.csv")]
imdir = os.path.join(cwd, 'figures', 'temp')


def log_image_to_wandb(fig, name:str, dir:str):
    impath = os.path.join(dir, f"{name}.png")
    fig.savefig(impath, **plot_kwargs)
    wandb.log({name: wandb.Image(impath)})


def main(config):
    train, test, train_images, test_images, gev_params = tf_utils.load_era5_datasets(roots, config.train_size, config.batch_size, im_size, paddings=paddings, conditions=conditions, viz=False)
    # _, _, orig_images, _ = tf_utils.load_era5_datasets(roots, config.train_size, config.batch_size, im_size, paddings=paddings, conditions=conditions, scale=True, viz=False)

    # train test callbacks
    chi_score = DCGAN.ChiScore({'train': next(iter(train)), 'test': next(iter(test))}, frequency=config.chi_frequency)
    cross_entropy = DCGAN.CrossEntropy(next(iter(test)))

    import pdb; pdb.set_trace()
    # compile
    with tf.device('/gpu:0'):
        gan = DCGAN.compile_dcgan(config)
        gan.fit(train, epochs=config.nepochs, callbacks=[WandbCallback(), chi_score, cross_entropy])

    finish_time = datetime.now().strftime("%Y%m%d")
    gan.generator.save_weights(os.path.join(wd, 'saved_models', f'{finish_time}_generator_weights'))
    gan.discriminator.save_weights(os.path.join(wd, 'saved_models', f'{finish_time}_discriminator_weights'))

    # generate 1000 images to visualise some results
    synthetic_data = gan(1000)
    synthetic_data = tf_utils.tf_unpad(synthetic_data, paddings).numpy()

    fig = viz_utils.plot_generated_marginals(synthetic_data)
    log_image_to_wandb(fig, 'generated_marginals', imdir)

    fig = viz_utils.compare_ecs_plot(train_images, test_images, synthetic_data, orig_images, channel=0)
    log_image_to_wandb(fig, 'correlations_u10', imdir)

    fig = viz_utils.compare_ecs_plot(train_images, test_images, synthetic_data, orig_images, channel=1)
    log_image_to_wandb(fig, 'correlations_v10', imdir)

    fig = viz_utils.compare_channels_plot(train_images, test_images, synthetic_data)
    log_image_to_wandb(fig, 'correlations multivariate', imdir)


if __name__ == "__main__":
    wandb.init(settings=wandb.Settings(code_dir="."))

    rundir = os.path.join(wd, "saved-models", wandb.run.name)
    os.makedirs(rundir)

    # tf.keras.utils.set_random_seed(wandb.config['seed'])  # sets seeds for base-python, numpy and tf
    tf.config.experimental.enable_op_determinism()  # removes stochasticity from individual operations
    main(wandb.config)
