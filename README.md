## Compressing Word Embeddings Via Deep Compositional Code Learning (Pytorch Implementation)

This repository contains my personal implementation of the following work:

Shu, R., Nakayama, H. (2018). [Compressing Word Embeddings Via Deep Compositional Code Learning](https://arxiv.org/pdf/1711.01068.pdf).

I was able to implement their compositional code learning method, evaluate on sentiment transfer tasks, and compare word similarity between codes.
Evaluation on machine translation tasks has not been implemented yet.

## Dependencies
* Python 3
* Pytorch (version 0.4.0)
* Torchtext
* Numpy
* GloVe vectors (Download glove.42B.300d.zip from https://nlp.stanford.edu/projects/glove/)

## How to use
To construct embeddings and other text files necessary for learning codes (only needs to be run once):
```
python3 construct_embeddings.py
```
To train the model to generate codes and embeddings (look inside train_code_learner.py for full list of commands):
```
python3 train_code_learner.py
```
To train a classifier on GloVe embeddings or compositional coded embeddings (check file for commands):
```
python3 train_classifier.py
```
To generate codes for words:
```
python3 code_analysis.py
```
