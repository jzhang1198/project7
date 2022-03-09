# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike


# Defining preprocessing functions
def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one hot encoding of a list of nucleic acid sequences
    for use as input into a fully connected neural net.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence
            length due to the one hot encoding scheme for nucleic acids.

            For example, if we encode
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    """
    one_hot_dict = {'A' : [1, 0, 0, 0],
                    'T' : [0, 1, 0, 0],
                    'C' : [0, 0, 1, 0],
                    'G' : [0, 0, 0, 1]}

    encodings = []
    for seq in seq_arr: #iterate through sequences in seq_arr
        one_hot_encoded = [] #generate an empty list to hold one hot encodings for bases
        for base in seq: #iterate through all bases in seq
            one_hot_encoded.append(one_hot_dict[base]) #substitute base for one hot encoding
        encodings.append([item for sublist in one_hot_encoded for item in sublist]) #flatten list of lists

    return encodings

def sample_seqs(seqs, labels, seed=None):
    """
    This function should sample your sequences to account for class imbalance.
    Consider this as a sampling scheme with replacement.

    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[int]
            List of positive/negative labels, where a label of 1 indicates a positive example and a label of 0 indicates a negative example

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[int]
            List of labels for the sampled sequences
    """

    if seed is not None:
        np.random.seed(seed)

    no_positives = labels.count(1)

    sampled_seqs = [seq for seq in seqs if labels[seqs.index(seq)] == 1]
    sampled_labels = [label for label in labels if label == 1]

    negative_indices = [index for index, element in enumerate(labels) if element == 0]
    for i in range(0,no_positives):
        rand_index = np.random.randint(0,len(negative_indices)-1) #pick a random index for a negative example
        assert negative_indices[rand_index] > 136
        rand_seq = seqs[negative_indices[rand_index]] #get sequence corresponding to index
        rand_label = labels[rand_index] #get label corresponding to index
        rand_pos = np.random.randint(0,len(rand_seq)-16) #pick a random starting position within negative seq
        rand_17mer = rand_seq[rand_pos:rand_pos+17] #extract 17mer

        sampled_seqs.append(rand_17mer)
        sampled_labels.append(rand_label)

    return sampled_seqs, sampled_labels
