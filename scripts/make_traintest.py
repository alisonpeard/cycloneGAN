"""
Processes ERA5 data.

Separates data into train and test of specified size. Fits GEV distribution to all pixels in train
and test separately. Output train and test set, GEV parameters, and train and test set transformed
to marginals. Additionally, given cyclone_size N, calculates the number of cyclone
days in the train set M and grabs N - M more cyclones from the test set, creating a
separate training set of N cyclones for transfer learning. The cyclone train set is
marginalised using the GEV parameters from the original training set.
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from evtGAN import DCGAN, tf_utils, viz_utils

# set scenario using command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--train-size', type=int, help='Size of train set', default='200')
parser.add_argument('-c', '--cyclone-size', type=int, help='Size of cyclone train set', default='50')
args = parser.parse_args()
train_size = args.train_size
cyclone_size = args.cyclone_size
root = "/Users/alison/Documents/DPhil/multivariate/wind_data/"

# some static variables
if __name__ == "__main__":
    for dim in ["u10", "v10"]:
        inpath = os.path.join(root, f"{dim}_dailymax.csv")
        df = pd.read_csv(inpath).sample(frac=1).reset_index(drop=True)  # shuffle
        n = len(df)
        assert train_size < n, f"Requested train size ({train_size}) exceeds size of dataset ({n})."

        # calculate how many cyclones are in train and take more from test set for more training
        cyclone_inds = df[df['cyclone_flag']].index
        normal_inds = df[~df['cyclone_flag']].index
        cyclone_train = [x for x in cyclone_inds if x < train_size]
        cyclone_test = [x for x in cyclone_inds if x >= train_size]
        m = cyclone_size - len(cyclone_train)
        cyclone_train = cyclone_train + cyclone_test[:m]
        cyclone_test = cyclone_test[m:]

        df = df.drop(columns=['cyclone_flag', 'time'])
        vals = df.values.astype(float)

        # fit GEV to entire train and test data
        train = vals[:train_size, :]
        test = vals[train_size:, :]
        cyclone_train = vals[cyclone_train]
        cyclone_test = vals[cyclone_test]

        train = tf.image.resize(train.reshape([len(train), 61, 61, 1]), [18, 22]).numpy().reshape([len(train), 18 * 22])
        test = tf.image.resize(test.reshape([len(test), 61, 61, 1]), [18, 22]).numpy().reshape([len(test), 18 * 22])
        cyclone_train = tf.image.resize(cyclone_train.reshape([len(cyclone_train), 61, 61, 1]), [18, 22]).numpy().reshape([len(cyclone_train), 18 * 22])
        cyclone_test = tf.image.resize(cyclone_test.reshape([len(cyclone_test), 61, 61, 1]), [18, 22]).numpy().reshape([len(cyclone_test), 18 * 22])

        # transform to marginals
        train_params = tf_utils.fit_gev(train)  # takes some time
        test_params = tf_utils.fit_gev(test)
        marginals_train = tf_utils.gev_marginals(train, train_params)
        marginals_test = tf_utils.gev_marginals(test, test_params)

        cyclone_marginals_train = tf_utils.gev_marginals(cyclone_train, train_params)
        cyclone_quantiles_train = tf_utils.gev_quantiles(cyclone_marginals_train, train_params)
        cyclone_marginals_test = tf_utils.gev_marginals(cyclone_test, test_params)

        # have a look
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(cyclone_train[0, ...].reshape([18, 22]))
        axs[1].imshow(cyclone_marginals_train[0, ...].reshape([18, 22]))
        axs[2].imshow(cyclone_quantiles_train[0, ...].reshape([18, 22]))
        axs[0].set_title('original')
        axs[1].set_title('marginals')
        axs[2].set_title('re-transformed')
        plt.show()

        # save them
        outdir = os.path.join(root, "..", "processed_wind_data",  f"train_{train_size}")
        np.save(os.path.join(outdir, f"cyclone_original_{dim}_train.npy"), cyclone_train)
        np.save(os.path.join(outdir, f"cyclone_original_{dim}_test.npy"), cyclone_test)
        np.save(os.path.join(outdir, f"cyclone_marginals_{dim}_train.npy"), cyclone_marginals_train)
        np.save(os.path.join(outdir, f"cyclone_marginals_{dim}_test.npy"), cyclone_marginals_test)
        np.save(os.path.join(outdir, f"original_{dim}_train.npy"), train)
        np.save(os.path.join(outdir, f"original_{dim}_test.npy"), test)
        np.save(os.path.join(outdir, f"marginals_{dim}_train.npy"), marginals_train)
        np.save(os.path.join(outdir, f"marginals_{dim}_test.npy"), marginals_test)
        np.save(os.path.join(outdir, f"gev_params_{dim}_train.npy"), np.array(train_params))
        np.save(os.path.join(outdir, f"gev_params_{dim}_test.npy"), np.array(train_params))
