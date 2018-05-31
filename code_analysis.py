import torch
import numpy as np
import argparse
from models import Code_Learner

# Use the GPU if it's available
use_gpu = torch.cuda.is_available()
parser = argparse.ArgumentParser(description='Generates codes for words')
# Directory where we want to write everything we save in this script to
parser.add_argument('--data_folder', default='data/', metavar='DIR',
                    help='folder to retrieve embeddings, data, text files, etc.')
parser.add_argument('--embedding_size', default=300, type=int, metavar='N', help='Embedding dimension size, default: 300')
parser.add_argument('--M', default=64, type=int, metavar='N', help='Number of source dictionaries, default: 64')
parser.add_argument('--K', default=8, type=int, metavar='N', help='Source dictionary size, default: 8')
parser.add_argument('--model_file', default='models/64_8/epoch_1500.pt', metavar='DIR',help='specific directory to model you want to load')
parser.add_argument('--words', default=['dog', 'cat'], metavar='[word1 ,word2]',help='words to find codes for')


def compare_codes(model, glove_dict, word1, word2):
    # Pass GloVe vectors into encoders to get codes
    vec1, vec2 = glove_dict[word1], glove_dict[word2]
    if use_gpu:
        vec1, vec2 = vec1.cuda(), vec2.cuda()
    _, code1 = model.encoder(vec1, training=False)
    _, code2 = model.encoder(vec2, training=False)
    print(word1, code1)
    print(word2, code2)
    return code1, code2


def main():
    global args
    # Parse commands from ArgumentParser
    args = parser.parse_args()
    # Load GloVE embeddings
    orig_embeddings = torch.load(args.data_folder + 'all_orig_emb.pt')
    # Load all GloVE words
    with open(args.data_folder + "glove_words.txt", "r") as file:
        glove_words = file.read().split('\n')
    # Recreate GloVE_dict
    glove_dict = {}
    for i, word in enumerate(glove_words):
        glove_dict[word] = orig_embeddings[i]
    # Load up the Code Learner model
    model = Code_Learner(args.embedding_size, args.M, args.K)
    model = torch.load(args.model_file)
    if use_gpu:
        model = model.cuda()
    # Generate codes
    compare_codes(model, glove_dict, *args.words)

if __name__ == '__main__':
    main()
