from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
# --
import argparse
import numpy as np
import os
import stats
import tensorflow as tf
from model import Model
from data import Dataset


RESULTS_DIRECTORY = "results"


class Config(object):
    """Set up model for debugging."""

    def __init__(self, data_path, model_name):
        self.data_path = data_path

        self.logdir = "../models/" + model_name + "/logs"
        self.ckptdir = "../models/" + model_name + "/ckpts"

        self.batch_size = 32
        self.current_epoch = 0
        self.num_epochs = 100
        self.feature_dim = 39
        self.num_layers = 3
        self.hidden_size = 256
        self.bidirectional = True
        self.keep_prob = 0.7
        self.margin = 0.5
        self.max_same = 1
        self.max_diff = 5
        self.lr = 0.001
        self.mom = 0.9
        self.log_interval = 10
        self.ckpt = None
        self.debugmode = True

        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.ckptdir, exist_ok=True)


def save(AUC, pivot, path, name):
    """
    Save AUC and ROC pivot points to the results directory. AUC is stored in a txt 
    file and ROC pivot points are saved in a csv file.

    :param AUC: float (Area Under the Curve)
    :param pivot: pandas data frame pivot representation
    :param path: results path
    :param name: filename prefix
    :return: None
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    with open(os.path.join(path, name + "_auc.txt"), 'w+') as f:
        f.write("%s\n" % AUC)
    pivot.to_csv(os.path.join(path, name + "_pivots.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description='Neural Acoustic Word Embeddings')
    parser.add_argument('-t', '--trimdata', type=float, default=1.0, help='Enable trimming of test, validation and training lists to the given percentage')
    parser.add_argument('modelname')
    parser.add_argument('datapath')
    args = parser.parse_args()

    config = Config(args.datapath, args.modelname)

    train_data = Dataset(partition="train", config=config, trim_data_percentage=args.trimdata)
    test_data = Dataset(partition="test", config=config, trim_data_percentage=args.trimdata)

    train_model = Model(is_train=True, config=config, reuse=None)
    test_model = Model(is_train=False, config=config, reuse=True)

    batch_size = config.batch_size

    saver = tf.train.Saver()

    proto = tf.ConfigProto(intra_op_parallelism_threads=2)
    with tf.Session(config=proto) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(config.ckptdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restored from %s" % ckpt.model_checkpoint_path)

        test_embeddings, test_labels = [], []
        for x, ts, ids in test_data.batch(batch_size):
            test_embeddings.append(test_model.get_embeddings(sess, x, ts))
            test_labels.append(ids)
        test_embeddings, test_labels = np.concatenate(test_embeddings), np.concatenate(test_labels)

        train_embeddings, train_labels = [], []
        for x, ts, ids in train_data.batch(batch_size):
            train_embeddings.append(train_model.get_embeddings(sess, x, ts))
            train_labels.append(ids)
        train_embeddings, train_labels = np.concatenate(train_embeddings), np.concatenate(train_labels)

        resultsPath = os.path.join(RESULTS_DIRECTORY, "nawe_" + args.modelname)

        AUC, pivot = stats.compute_ROC_curve(test_embeddings, train_embeddings)
        save(AUC, pivot, resultsPath, "test")

if __name__ == "__main__":
    main()
