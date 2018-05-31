## Compressing Word Embeddings Via Deep Compositional Code Learning (Pytorch Implementation)

This repository contains my personal implementation of the following work:

Shu, R., Nakayama, H. (2018). [Compressing Word Embeddings Via Deep Compositional Code Learning](https://arxiv.org/pdf/1711.01068.pdf).

I was able to implement their compositional code learning method, evaluate on sentiment classification tasks, and compare word similarity between codes.
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

## Brief Results
Using a 8x8 coding model, I got the following encodings:

|             |                 |
| ------------- | ------------- |
| cat       	|  5,6,4,7,5,5,3,6 	|
| dog       	|  0,6,4,7,4,0,3,6 	|
| cow       	|  0,5,4,7,4,5,3,6 	|
|           	|                   |
| blue      	|  5,6,4,7,5,5,3,6 	|
| purple    	|  5,6,4,7,4,5,3,2 	|
|           	|                   |
| president 	|  4,6,4,7,5,1,4,6  |
| governor  	|  0,6,4,7,5,1,4,6  |

Words that are similar have very similar encodings. Also, for sentiment classification:

|      Model       |          Accuracy     |
| ------------- | ------------- |
| Classifier with baseline GloVe embedding | 0.853|
| Classifier with 64x8 encoding | 0.841|

Therefore, I was able to replicate the results of the paper fairly well, and I also received very similar accuracy scores for the two classifiers, even though the the 64x8 encoding classifier is 3% the size of the baseline classifier.
