import numpy as np
import re
import itertools
from collections import Counter
from konlpy.tag import Mecab


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_data_and_labels2(file_name):
    positive_exams = []
    negative_exams = []
    positive_count = 0
    negative_count = 0

    exams = list(open(file_name, "r").readlines())
    for s in exams:
        splited = s.split('\t')
        if splited[2] == '0\n':
            negative_exams.append(splited[1])
            negative_count = negative_count + 1
        elif splited[2] == '1\n':
            positive_exams.append(splited[1])
            positive_count = positive_count + 1
        else:
            print (splited[0], splited[1], splited[2])

    mecab = Mecab()

    positive_result = []
    for pp in positive_exams:
        one_str =  mecab.pos(pp)
        str_result = ''
        for p in one_str:
            if p[1] in {'NNG', 'NNP', 'NNB', 'NNBC', 'VA', 'VV', 'SL', 'SN', 'SY'}:
                str_result = p[0] + ' ' + str_result
        positive_result.append(str_result)

    positive_labels = [[0, 1] for _ in positive_result]

    negative_result = []
    for pp in negative_exams:
        one_str =  mecab.pos(pp)
        str_result = ''
        for p in one_str:
            if p[1] in {'NNG', 'NNP', 'NNB', 'NNBC', 'VA', 'VV', 'SL', 'SN', 'SY'}:
                str_result = p[0] + ' ' + str_result
        negative_result.append(str_result)

    negative_labels = [[1, 0] for _ in negative_result]

    y = np.concatenate([positive_labels, negative_labels], 0)

    x_text = positive_result + negative_result
        
    return [x_text, y]

# data : x_train, y_train 의 zip 형태
# batch_size : 64
# num_epochs : 200
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    print (" data_size : ", data_size)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            print (" num_batches_per_epoch : ", num_batches_per_epoch)
            print (" batch_num, batch_size : ", batch_num, batch_size)
            print ("start_index, end_index : ", start_index, end_index)
            print ("shuffle_data : ", shuffled_data[start_index:end_index])
            yield shuffled_data[start_index:end_index]
