import torch
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu

from vocabulary import Vocabulary
from config import *
import string



def read_lines(filepath):
    """ Open the ground truth captions into memory, line by line. 
    Args:
        filepath (str): the complete path to the tokens txt file
    """
    file = open(filepath, 'r')
    lines = []

    while True: 
        # Get next line from file 
        line = file.readline() 
        if not line: 
            break
        lines.append(line.strip())
    file.close()
    return lines
    #print(lines)


def parse_lines(lines):
    """
    Parses token file captions into image_ids and captions.
    Args:
        lines (str list): str lines from token file
    Return:
        image_ids (int list): list of image ids, with duplicates
        cleaned_captions (list of lists of str): lists of words
    """
    table = str.maketrans('', '', string.punctuation)
    image_ids = []
    cleaned_captions = []
    for ln in lines:
        tkn = ln.split()
        im_id, im_cap = tkn[0], tkn[1:]
        im_id = im_id.split('.')[0]
        #clean captions like punctuation,comma,numeric words,convert to lowercase etc
        im_cap = [word.lower() for word in im_cap]
        im_cap = [w.translate(table) for w in im_cap]
        im_cap = [word for word in im_cap if word.isalpha()]
        im_cap = ' '.join(im_cap)
        image_ids.append(im_id)
        cleaned_captions.append(im_cap)


    return image_ids, cleaned_captions


def build_vocab(cleaned_captions):
    """ 
    Parses training set token file captions and builds a Vocabulary object
    Args:
        cleaned_captions (str list): cleaned list of human captions to build vocab with

    Returns:
        vocab (Vocabulary): Vocabulary object
    """

    # QUESTION 1.1
    # TODO collect words


    # create a vocab instance
    vocab = Vocabulary()

    # add the token words
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # TODO add the rest of the words from the cleaned captions here
    # vocab.add_word('word')
    cap_list = []
    cap_freq = []
    for cap in cleaned_captions:
        cap_set=cap.split()
        for item in cap_set:
            cap_list.append(item)
	#get frequent words
    for cap1 in cap_list:
        if cap_list.count(cap1)>3:
            cap_freq.append(cap1)
	#add words to vocab
    for word in set(cap_freq):
        vocab.add_word(word)

    return vocab


def decode_caption(sampled_ids, vocab):
    """ 
    Args:
        sampled_ids (int list): list of word IDs from decoder
        vocab (Vocabulary): vocab for conversion
    Return:
        predicted_caption (str): predicted string sentence
    """
	    
    #predicted_caption = ""
    # QUESTION 2.1
    sampled_caption=[]
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        #id_word=vocab.word2idx[word]
        #print(word)
        sampled_caption.append(word)
        if word == '<end>':
            break
    predicted_caption = ' '.join(sampled_caption)

    return predicted_caption


"""
We need to overwrite the default PyTorch collate_fn() because our 
ground truth captions are sequential data of varying lengths. The default
collate_fn() does not support merging the captions with padding.

You can read more about it here:
https://pytorch.org/docs/stable/data.html#dataloader-collate-fn. 
"""
def caption_collate_fn(data):
    """ Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 224, 224).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 224, 224).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length from longest to shortest.
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # merge images (from tuple of 3D tensor to 4D tensor).
    # if using features, 2D tensor to 3D tensor. (batch_size, 256)
    images = torch.stack(images, 0) 

    # merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths