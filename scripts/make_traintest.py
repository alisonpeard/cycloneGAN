import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from evtGAN import DCGAN, tf_utils, viz_utils

# set scenario using command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--train-size', type=int, help='Size of train set', default='50')
args = parser.parse_args()
train_size = args.train_size
root = "/Users/alison/Documents/DPhil/multivariate/wind_data/"

# some static variables
if __name__ == "__main__":
    for dim in ["u10", "v10"]:
        inpath = os.path.join(root, f"{dim}_dailymax.csv")
        df = pd.read_csv(inpath).sample(frac=1).reset_index(drop=True)  # shuffle
        n = len(df)
        assert train_size < n, f"Requested train size ({train_size}) exceeds size of dataset ({n})."

        cyclone_inds = df[df['cyclone_flag']].index
        normal_inds = df[~df['cyclone_flag']].index
        cyclone_train = [x for x in cyclone_inds if x < train_size]
        cyclone_test = [x for x in cyclone_inds if x >= train_size]
        normal_train = [x for x in normal_inds if x < train_size]
        normal_test = [x for x in normal_inds if x >= train_size]

        df = df.drop(columns=['cyclone_flag', 'time'])
        vals = df.values.astype(float)

        # fit GEV to entire train and test data
        train_set = vals[:train_size, :]
        test_set = vals[train_size:, :]
        train_params = tf_utils.fit_gev(train_set)  # takes some time
        test_params = tf_utils.fit_gev(test_set)

        cyclone_train = vals[cyclone_train]
        cyclone_test = vals[cyclone_test]
        normal_train = vals[normal_train]
        normal_test = vals[normal_test]

        # transform to marginals
        cyclone_marginals_train = tf_utils.gev_marginals(cyclone_train, train_params)
        cyclone_marginals_test = tf_utils.gev_marginals(cyclone_test, test_params)
        cyclone_quantiles_train = tf_utils.gev_quantiles(cyclone_marginals_train, train_params)
        normal_marginals_train = tf_utils.gev_marginals(normal_train, train_params)
        normal_marginals_test = tf_utils.gev_marginals(normal_test, test_params)

        # have a look
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(cyclone_train[0, ].reshape([61, 61]))
        axs[1].imshow(cyclone_marginals_train[0, ].reshape([61, 61]))
        axs[2].imshow(cyclone_quantiles_train[0, ].reshape([61, 61]))
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
        np.save(os.path.join(outdir, f"normal_original_{dim}_train.npy"), normal_train)
        np.save(os.path.join(outdir, f"normal_original_{dim}_test.npy"), normal_test)
        np.save(os.path.join(outdir, f"normal_marginals_{dim}_train.npy"), normal_marginals_train)
        np.save(os.path.join(outdir, f"normal_marginals_{dim}_test.npy"), normal_marginals_test)
        np.save(os.path.join(outdir, f"gev_params_{dim}_train.npy"), np.array(train_params))
        np.save(os.path.join(outdir, f"gev_params_{dim}_test.npy"), np.array(train_params))
