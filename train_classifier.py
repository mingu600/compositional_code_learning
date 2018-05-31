import torchtext
import os
import torch.optim as optim
import torch
import numpy as np
import math
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models import Code_Learner, Classifier

# Use the GPU if it's available
use_gpu = torch.cuda.is_available()
parser = argparse.ArgumentParser(description='Train a classifier to compare performance between GloVE and encoding')
# Directory where we want to write everything we save in this script to
parser.add_argument('--data_folder', default='data/', metavar='DIR',
                    help='folder to retrieve embeddings, data, text files, etc.')
parser.add_argument('--models_folder', default='models/', metavar='DIR',help='folder to save models')
parser.add_argument('--model_file', default='models/64_8/epoch_7000.pt', metavar='DIR',help='specific directory to model you want to load')
parser.add_argument('--embedding_size', default=300, type=int, metavar='N', help='Embedding dimension size, default: 300')
parser.add_argument('--M', default=64, type=int, metavar='N', help='Number of source dictionaries, default: 64')
parser.add_argument('--K', default=8, type=int, metavar='N', help='Source dictionary size, default: 8')
parser.add_argument('--lr', default=0.0001, type=float, metavar='N', help='Adam learning rate, default: 0.0001')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='Minibatch size, default: 128')
parser.add_argument('--max_len', default=400, type=int, metavar='N', help='Max sentence length allowed, default: 400')
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='Total number of epochs, default: 30')
parser.add_argument('--embedding_type', default='coded', metavar='MODEL_TYPE', help='Model type; pick either coded or baseline, default: coded')


def classifier_train(epochs, classifier, optimizer, loss_func, train_iter, test_iter, c_type):
    classifier.train()
    # Initialize validation loss
    best_val_loss = float('inf')
    # For every epoch
    for epoch in range(epochs):
        valid_loss = 0
        # For every batch
        for batch in train_iter:
            # Each batch has review, label
            data, label = batch.text, batch.label
            if use_gpu:
                data, label = data.cuda(), label.cuda()
            # Labels in original dataset are given as 1 and 2, so we make that 0 and 1 instead
            label = label - 1
            # Clear gradients
            optimizer.zero_grad()
            # Make predictions with our _classifier
            preds = classifier(data)
            # Calculate loss
            loss = loss_func(preds, label)
            # Compute sum of gradients
            loss.backward()
            # Perform optimization step
            optimizer.step()

        # VALIDATION
        correct = 0
        total_val_examples = 0
        # Go through same steps as training, except do not update weights
        for batch in test_iter:
            valid_data, valid_label = batch.text, batch.label
            if use_gpu:
                valid_data, valid_label = valid_data.cuda(), valid_label.cuda()
            valid_label = valid_label - 1
            # Give hard predictions instead of soft ones
            valid_preds = classifier(valid_data).data.max(1)[1]
            # Count how many the model got correct
            correct += torch.sum(valid_preds == valid_label).item()
            # Update number of total examples
            total_val_examples += len(valid_label)
        # If this is our lowest validation loss, save the model
        if valid_loss < best_val_loss:
            torch.save(classifier, args.model_folder + c_type + '_classifier/epoch_' + str(epoch) + '.pt')
            best_val_loss = valid_loss
        # Calculate accuracy and report
        print('''Epoch [{e}/{num_e}]\t Accuracy: {r_l:.3f}'''.format(e=epoch+1, num_e=epochs, r_l = correct / total_val_examples))

def main():
    global args
    # Parse commands from ArgumentParser
    args = parser.parse_args()
    # Our text field for imdb data
    TEXT = torchtext.data.Field(lower=True)
    # Our label field for imdb data
    LABEL = torchtext.data.Field(sequential=False)
    # Load GloVE embeddings
    orig_embeddings = torch.load(args.data_folder + 'all_orig_emb.pt')
    total_words = len(orig_embeddings)
    # Load shared words and all GloVE words
    with open(args.data_folder + "shared_words.txt", "r") as file:
        shared_words = file.read().split('\n')
    with open(args.data_folder + "glove_words.txt", "r") as file:
        glove_words = file.read().split('\n')
    # Recreate GloVE_dict
    glove_dict = {}
    for i, word in enumerate(glove_words):
        glove_dict[word] = orig_embeddings[i]

    # Load IMDB dataset with standard splits and restrictions identical to paper
    train, test = torchtext.datasets.IMDB.splits(TEXT, LABEL, filter_pred=lambda ex: ex.label != 'neutral' and len(ex.text) <= 400)

    # Both loops go through the words of train and test dataset, finds words without glove vectors, and replaces them with <unk>
    for i in range(len(train)):
        review = train.examples[i].text
        for i, word in enumerate(review):
            if word not in glove_dict:
                review[i] = '<unk>'
    for i in range(len(test)):
        review = test.examples[i].text
        for i, word in enumerate(review):
            if word not in glove_dict:
                review[i] = '<unk>'

    # Build modified vocabulary
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    # Create iterators over train and test set
    train_iter, test_iter = torchtext.data.BucketIterator.splits((train, test), batch_size=args.batch_size, repeat=False, device=-1)

    # If we want to use baseline GloVE embeddings
    if args.embedding_type == 'baseline':
        # Initialize embedding
        comp_embedding = np.random.uniform(-0.25, 0.25, (len(TEXT.vocab), args.embedding_size))
        # For each vocab word, replace embedding vector with GloVE vector
        for word in shared_words:
            comp_embedding[TEXT.vocab.stoi[word]] = glove_dict[word]
        # Initialize Classifer with our GloVE embedding
        base_c = Classifier(torch.FloatTensor(comp_embedding), args.batch_size)
        # Put model into CUDA memory if using GPU
        if use_gpu:
            base_c = base_c.cuda()
        # Initialize Optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,base_c.parameters()), lr=args.lr)
        # Define Loss function
        loss_func = nn.NLLLoss()

    else:
        '''
        Note- the model in the paper is different because they only store the source dictionaries,
        making their model smaller than normal classifiers which is a major purpose of the paper.
        By my formulation, my model actually has the same size. However, they are fundamentally equivalent,
        except that the authors would have to preprocess the data (convert words into codes) whereas I
        simply make an embedding layer of size Vocab like GloVE vectors. Either way, I should get the same
        levels of accuracy, which is the primary importance of the sentiment classification task- to check
        whether the coding embeddings still give the same level of accuracy.
        '''
        # Initialize embedding
        code_embedding = torch.FloatTensor(np.random.uniform(-0.25, 0.25, (len(TEXT.vocab), args.embedding_size)))
        # Load best model for code embedding generation
        model = Code_Learner(args.embedding_size, args.M, args.K)
        model = torch.load(args.model_file)
        # Put model into CUDA memory if using GPU
        if use_gpu:
            code_embedding = code_embedding.cuda()
            model = model.cuda()
        # For all words in vocab
        for i in range(len(TEXT.vocab)):
            # Try to see if it has a corresponding glove_vector
            try:
                glove_vec = glove_dict[TEXT.vocab.itos[i]]
                if use_gpu:
                    glove_vec = glove_vec.cuda()
                # If so, then generate our own embedding for the word using our model
                code_embedding[i] = model(glove_vec, training=False)
            # The word doesn't have a GloVE vector, keep it randomly initialized
            except KeyError:
                pass
        base_c = Classifier(torch.FloatTensor(code_embedding.cpu()), args.batch_size)
        # Put model into CUDA memory if using GPU
        if use_gpu:
            base_c = base_c.cuda()
        # Initialize Optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,base_c.parameters()), lr=args.lr)
        # Define Loss function
        loss_func = nn.NLLLoss()

    classifier_train(args.epochs, base_c, optimizer, loss_func, train_iter, test_iter, args.embedding_type)

if __name__ == '__main__':
    main()
