from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import re, os
import numpy as np
from collections import Counter
from python_speech_features import delta
from python_speech_features import mfcc
import scipy.io.wavfile as wav

def build_training_list(path, testing_list, validation_list):
    """
    Based on SpeechCommand_v0.02 directory structure.
    """
    result = []
    walk_tuple = os.walk(path)
    directory_list = [x[1] for x in walk_tuple][0]
    directory_list = [x for x in directory_list if "_background_noise_" not in x]
    for directory in directory_list:
        file_list = os.listdir(path + directory)
        for wav_file in file_list:
            wav_small_path = directory + "/" + wav_file
            if wav_small_path not in testing_list and wav_small_path not in validation_list:
                result.append(wav_small_path)
    return result

def trim_data(data, percentage=0.3):
    """
    Based on SpeechCommand_v0.02 directory structure.
    """
    length_list = []
    current_directory_length = 0
    current_directory = data[0].split("/")[-2]
    for element in data:
        element_directory = element.split("/")[-2]
        if element_directory != current_directory:
            length_list.append(current_directory_length)
            current_directory = element_directory
            current_directory_length = 1
        else:
            current_directory_length += 1
    length_list.append(current_directory_length)

    reduced_length_list = []
    for length in length_list:
        reduced_length_list.append(int(length * percentage))
    
    result = []
    current_directory_index = 0
    current_directory = data[0].split("/")[-2]
    cpt = 0
    for i, element in enumerate(data):
        element_directory = element.split("/")[-2]
        if element_directory != current_directory:
            current_directory = element_directory
            current_directory_index += 1
            cpt = 1
        else:
            if cpt < reduced_length_list[current_directory_index]:
                result.append(data[i])
            cpt += 1
    return result

def get_data(path, partition, trim_data_percentage=1.0):
    if partition == "train":
        with open(path + "testing_list.txt") as f:
            testing_list = f.read().splitlines()
        with open(path + "validation_list.txt") as f:
            validation_list = f.read().splitlines()
        wav_list = build_training_list(path, testing_list, validation_list)
    elif partition == "dev":
        with open(path + "validation_list.txt") as f:
            wav_list = f.read().splitlines()
    elif partition == "test":
        with open(path + "testing_list.txt") as f:
            wav_list = f.read().splitlines()

    if trim_data_percentage < 1.0:
        wav_list = trim_data(wav_list, trim_data_percentage)
    
    data = []
    labels = []
    for elt in wav_list:
        (rate, signal) = wav.read(path+elt)
        mfcc_static = mfcc(signal, rate)
        mfcc_deltas = delta(mfcc_static, 2)
        mfcc_delta_deltas = delta(mfcc_deltas, 2)
        features = np.hstack([mfcc_static, mfcc_deltas, mfcc_delta_deltas])

        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        data.append(features)
        labels.append(elt)
    return labels, data

class Dataset(object):
    """Creat data class."""

    def __init__(self, partition, config, feature_mean=None, trim_data_percentage=1.0):
        """Initialize dataset."""
        self.is_train = (partition == "train")

        self.feature_dim = config.feature_dim

        data_path = getattr(config, "data_path")
        labels, data = get_data(data_path, partition, trim_data_percentage)

        words = [re.split("/", x)[0] for x in labels]
        uwords = np.unique(words)

        word2id = {v: k for k, v in enumerate(uwords)}
        ids = [word2id[w] for w in words]

        if feature_mean is None:
            feature_mean, n = 0.0, 0
            for x in data:
                feature_mean += np.sum(x)
                n += np.prod(x.shape)
            feature_mean /= n
        self.feature_mean = feature_mean

        self.data = np.array([x - self.feature_mean for x in data])
        self.ids = np.array(ids, dtype=np.int32)
        self.id_counts = Counter(ids)

        self.num_classes = len(self.id_counts)
        self.num_examples = len(self.ids)

    def shuffle(self):
        """Shuffle data."""

        shuffled_indices = np.random.permutation(self.num_examples)
        self.data = self.data[shuffled_indices]
        self.ids = self.ids[shuffled_indices]


    def pad_features(self, indices):
        """Pad acoustic features to max length sequence."""
        b = len(indices)
        lens = np.array([len(xx) for xx in self.data[indices]], dtype=np.int32)
        padded = np.zeros((b, max(lens), self.feature_dim))
        for i, (x, l) in enumerate(zip(self.data[indices], lens)):
            padded[i, :l] = x

        return padded, lens, self.ids[indices]


    def batch(self, batch_size, max_same=1, max_diff=1):
        """Batch data."""

        # self.shuffle()

        same = []
        for index, word_id in enumerate(self.ids):  # collect same samples
            indices = np.where(self.ids == word_id)[0]
            same.append(np.random.permutation(indices[indices != index])[:max_same])
        same = np.array(same)

        diff_ids = np.random.randint(0, self.num_classes, (self.num_examples, max_diff))
        diff_ids[diff_ids >= np.tile(self.ids.reshape(-1, 1), [1, max_diff])] += 1

        diff = np.full_like(diff_ids, 0, dtype=np.int32)
        for word_id, count in self.id_counts.items():  # collect diff samples
            indices = np.where(diff_ids == word_id)
            diff[indices] = np.where(self.ids == word_id)[0][np.random.randint(0, count, len(indices[0]))]

        get_batch_indices = lambda start: range(start, min(start + batch_size, self.num_examples))

        for indices in map(get_batch_indices, range(0, self.num_examples, batch_size)):

            if self.is_train:
                b = len(indices)

                same_partition = [np.arange(b)]  # same segment ids for anchors
                same_partition += [(b + i) * np.ones(len(x)) for i, x in enumerate(same[indices])]  # same segment ids for same examples
                same_partition += [(2 * b) + np.arange(max_diff * b)]  # same segment ids for diff examples
                same_partition = np.concatenate(same_partition)

                diff_partition = np.concatenate([i * np.ones(max_diff) for i in range(b)])  # diff segment ids for diff examples

                indices = np.concatenate((indices, np.hstack(same[indices]), diff[indices].flatten()))

                data, lens, _ = self.pad_features(indices)
                yield data, lens, same_partition, diff_partition

            else:
                yield self.pad_features(indices)
