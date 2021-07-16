Neural Acoustic Word Embeddings for Speech Commands v0.02 & DyLNet
==================================================================

Overview:
---------

This is a recipe for learning neural acoustic word embeddings for a subset of Speech Commands v0.02 & DyLNet. The models are explained in greater detail in [Settle & Livescu, 2016](https://arxiv.org/abs/1611.02550) as well as [Settle et al., 2017](https://arxiv.org/abs/1706.03818):

- S. Settle and K. Livescu, "Discriminative Acoustic Word Embeddings: Recurrent Neural Network-Based Approaches," in Proc. SLT, 2016.
- S. Settle, K. Levin, H. Kamper, and K. Livescu, "Query-by-Example Search with Discriminative Neural Acoustic Word Embeddings," in Proc. Interspeech, 2017.


Contents:
---------

code/
- python code to create, run, and save the model


Steps:
------

1. Ensure access to installed dependencies.
    - Python 3.6
    - Tensorflow 1.5 (and numpy/scipy)
    - [kaldi](https://github.com/kaldi-asr/kaldi)
    - [kaldi-io-for-python](https://github.com/vesis84/kaldi-io-for-python)

2. Clone repo.

3. Download [Speech Commands v0.02](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) corpus

4. Navigate to code directory and run "python main.py -t=0.05 -l=sc2 <corpus_dir>". This will train, evaluate, and save the model named "sc2" on 5% of the corpus.
