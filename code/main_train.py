from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
# --
from os import path, makedirs
import argparse
import numpy as np
import tensorflow as tf
from model import Model
from data import Dataset
from average_precision import average_precision

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

        makedirs(self.logdir, exist_ok=True)
        makedirs(self.ckptdir, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Neural Acoustic Word Embeddings')
    parser.add_argument('-t', '--trimdata', type=float, default=1.0, help='Enable trimming of test, validation and training lists to the given percentage')
    parser.add_argument('modelname')
    parser.add_argument('datapath')
    args = parser.parse_args()

    config = Config(args.datapath, args.modelname)

    train_data = Dataset(partition="train", config=config, trim_data_percentage=args.trimdata)
    dev_data = Dataset(partition="dev", config=config, feature_mean=train_data.feature_mean, trim_data_percentage=args.trimdata)

    train_model = Model(is_train=True, config=config, reuse=None)
    dev_model = Model(is_train=False, config=config, reuse=True)

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

        batch = 0
        for epoch in range(config.current_epoch, config.num_epochs):
            print("epoch: ", epoch)

            losses = []
            for x, ts, same, diff in train_data.batch(batch_size, config.max_same, config.max_diff):
                _, loss = train_model.get_loss(sess, x, ts, same, diff)
                losses.append(loss)
                if batch % config.log_interval == 0:
                    print("avg batch loss: %.4f" % np.mean(losses[-config.log_interval:]))
                batch += 1

            embeddings, labels = [], []
            for x, ts, ids in dev_data.batch(batch_size):
                embeddings.append(dev_model.get_embeddings(sess, x, ts))
                labels.append(ids)
            embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
            print("ap: %.4f" % average_precision(embeddings, labels))

            saver.save(sess, path.join(config.ckptdir, "model"), global_step=epoch)

if __name__ == "__main__":
    main()
