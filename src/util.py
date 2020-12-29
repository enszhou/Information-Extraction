import nltk
import os
import string
import numpy as np
import tensorflow as tf
import time

relations = [
    "Cause-Effect",
    "Component-Whole",
    "Entity-Destination",
    "Product-Producer",
    "Entity-Origin",
    "Member-Collection",
    "Message-Topic",
    "Content-Container",
    "Instrument-Agency",
    "Other",
]
del_letters = string.punctuation
del_tran_table = str.maketrans(del_letters, " " * len(del_letters))
stopwords = set(nltk.corpus.stopwords.words("english"))

max_words = 20000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)

dataset_dir = os.path.join("..", "dataset", "")
train_path = os.path.join(dataset_dir, "train.txt")
test_path = os.path.join(dataset_dir, "test.txt")
train_text_path = os.path.join(dataset_dir, "train_text.txt")
train_label_path = os.path.join(dataset_dir, "train_label.txt")
test_text_path = os.path.join(dataset_dir, "test_text.txt")


def split_dataset():
    train_fp = open(train_path)
    train_text_fp = open(train_text_path, "w+")
    train_label_fp = open(train_label_path, "w+")
    while True:
        train_text = train_fp.readline()
        train_label = train_fp.readline()
        if not train_text or not train_label:
            break
        train_text = train_text.split(" ", 1)[1]
        train_text_fp.write(train_text)
        train_label_fp.write(train_label)

    test_fp = open(test_path)
    test_text_fp = open(test_text_path, "w+")
    while True:
        test_text = test_fp.readline()
        if not test_text:
            break
        test_text = test_text.split(" ", 1)[1]
        test_text_fp.write(test_text)


def tokenize(text):
    sentence = text.translate(del_tran_table).lower()
    tokens = nltk.tokenize.word_tokenize(sentence)
    return tokens


def filter_token(token):
    return token not in stopwords


def get_test_label_path():
    return os.path.join(
        "..",
        "output",
        "test_label_%s.txt" % time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
    )


def read_texts(path):
    texts = list()
    with open(path) as fp:
        while True:
            text = fp.readline()
            if not text:
                break
            # sentence = tokenize(text)
            # sentence = list(filter(filter_token, sentence))
            # sentence_pos = nltk.pos_tag(sentence)
            # sentence_pos = list(filter(lambda x: x[1][0]=='N', sentence_pos))
            # sentence = [x[0] for x in sentence_pos]
            texts.append(text)
    return texts


def read_labels(path):
    labels = list()
    with open(path) as fp:
        while True:
            label = fp.readline()
            if not label:
                break
            labels.append(label.split("(")[0])
    return labels


def write_labels(preds, path):
    with open(path, "w+") as fp:
        fp.write("\n".join(preds))


def one_hot(labels):
    Y = list(map(relations.index, labels))
    Y = np.eye(len(relations))[Y]
    return Y


def labelize(preds):
    preds = np.argmax(preds, axis=1)
    preds = list(map(lambda x: relations[x], preds))
    return preds


if __name__ == "__main__":
    split_dataset()
    # labels = get_labels(train_label_path)
    # Y = one_hot(labels)
    # preds = labelize(Y)
    # write_labels(preds, test_label_path)