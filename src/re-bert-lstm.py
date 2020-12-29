import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras import layers
import os
import sys
import matplotlib.pyplot as plt
from util import *

tf.get_logger().setLevel("ERROR")

# seq_length = 64
# lstm_nodes = 256
# valid_ratio = 0.1
# epochs = 10
seq_length = int(sys.argv[1])
lstm_nodes = int(sys.argv[2])
valid_ratio = float(sys.argv[3])
epochs = int(sys.argv[4])


texts = read_texts(train_text_path)
labels = read_labels(train_label_path)
X = tf.constant(texts)
Y = one_hot(labels)


preprocessor = hub.load(
    "https://hub.tensorflow.google.cn/tensorflow/albert_en_preprocess/2"
)
text_inputs = [tf.keras.layers.Input(shape=(), dtype=tf.string)]
tokenize = hub.KerasLayer(preprocessor.tokenize)
tokenized_inputs = [tokenize(segment) for segment in text_inputs]

bert_pack_inputs = hub.KerasLayer(
    preprocessor.bert_pack_inputs, arguments=dict(seq_length=seq_length)
)
encoder_inputs = bert_pack_inputs(tokenized_inputs)

encoder = hub.KerasLayer(
    "https://hub.tensorflow.google.cn/tensorflow/albert_en_base/2", trainable=False
)
outputs = encoder(encoder_inputs)
pooled_output = outputs["pooled_output"]
sequence_output = outputs["sequence_output"]


x = sequence_output
x = layers.Bidirectional(layers.LSTM(lstm_nodes, recurrent_dropout=0.2, dropout=0.2))(x)
outputs = layers.Dense(len(relations), activation="softmax")(x)
model = tf.keras.Model(inputs=text_inputs, outputs=outputs)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(X, Y, epochs=epochs, batch_size=32, validation_split=valid_ratio)


test_texts = read_texts(test_text_path)
X_test = tf.constant(test_texts)

preds = model.predict(X_test)
test_labels = labelize(preds)

write_labels(test_labels, get_test_label_path)