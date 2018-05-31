import torchtext
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Create GloVE and IMDB embeddings')
# File name for GloVE vectors
parser.add_argument('--glove_file', default='glove.42B.300d.txt', metavar='file',
                    help='file which contains GloVE embeddings')
# Directory where we want to write everything we save in this script to
parser.add_argument('--data_folder', default='data/', metavar='DIR',
                    help='folder to save embeddings, data, text files, etc.')

def main():
    global args
    # Parse commands from ArgumentParser
    args = parser.parse_args()
    # Our text field for imdb data
    TEXT = torchtext.data.Field(lower=True, fix_length=400)
    # Our label field for imdb data
    LABEL = torchtext.data.Field(sequential=False)

    # Use standard split for IMDB dataset, filtering out reviews that are longer than 400 words
    train, test = torchtext.datasets.IMDB.splits(TEXT, LABEL, \
    filter_pred=lambda ex: ex.label != 'neutral' and len(ex.text) <= 400)
    # Build vocabulary from training dataset
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    # Get a list of all words from imdb
    imdb_words = TEXT.vocab.freqs.keys()

    # Next, construct a dictionary that maps words to their GloVe vectors
    glove_dict = {}
    # There are this many words included in the file
    total_glove_num = 1917494
    # We also want to store all the embeddings to a file, which we can't do from a dict
    all_orig_embeddings = torch.zeros(total_glove_num, 300, dtype=torch.float)
    # We also want a list of all glove_words, because it is handy
    glove_words = []
    # Reading previously specified file
    with open(args.glove_file) as file:
        # For every line, the first part is the word, and the rest is the vector.
        for i, line in enumerate(file):
            entry = line.split()
            word = entry[0]
            embedding = np.array(entry[1:], dtype='float32')
            # Add word to our running list
            glove_words.append(word)
            # Add word -> embedding pair to dict
            glove_dict[word] = embedding
            # Also to our FloatTensor for all GloVe embeddings
            all_orig_embeddings[i] = torch.FloatTensor(embedding)
    print('GloVe dict constructed')

    # Now we make a list of words that appear in both the IMDB dataset and the GloVE file
    shared_words = []
    for word in imdb_words:
        if word in glove_dict:
            shared_words.append(word)
    print('Shared words list constructed.')
    # We write our shared_word list to a text file for easy reference
    with open(args.data_folder + 'shared_words.txt', 'w') as out_file:
        out_file.write('\n'.join(shared_words))

    # We write our glove_word list to a text file for easy reference
    with open(args.data_folder + 'glove_words.txt', 'w') as out_file:
        out_file.write('\n'.join(glove_words))

    # We save our glove_embedding for later use
    torch.save(all_orig_embeddings, args.data_folder + 'all_orig_emb.pt')

if __name__ == '__main__':
    main()
