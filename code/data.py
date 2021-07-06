from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import re, os
import numpy as np
from collections import Counter
from python_speech_features import fbank, lifter
from python_speech_features import delta
from scipy.fftpack import dct
import scipy.io.wavfile as wav

def calculate_nfft(samplerate, winlen):
    """
    source : https://github.com/jameslyons/python_speech_features/pull/76/files

    Calculates the FFT size as a power of two greater than or equal to
    the number of samples in a single window length.
    
    Having an FFT less than the window length loses precision by dropping
    many of the samples; a longer FFT than the window allows zero-padding
    of the FFT buffer which is neutral in terms of frequency domain conversion.
    :param samplerate: The sample rate of the signal we are working with, in Hz.
    :param winlen: The length of the analysis window in seconds.
    """
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft

def mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
         nfilt=26, nfft=None, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22,
         appendEnergy=True, winfunc=lambda x:np.ones((x,))):
    """
    source : https://github.com/jameslyons/python_speech_features/pull/76/files

    Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is None, which uses the calculate_nfft function to choose the smallest size that does not drop sample data.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    nfft = nfft or calculate_nfft(samplerate, winlen)
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
    feat = np.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = lifter(feat,ceplifter)
    if appendEnergy: feat[:,0] = np.log(energy) # replace first cepstral coefficient with log of frame energy
    return feat

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
