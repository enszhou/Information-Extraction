# Lab2信息抽取

## 目录说明

```
├── README.md							      // 本文件
├── dataset                                         // 数据集目录
│   ├── test.txt                                    // 原测试集
│   ├── test_text.txt                            // 预处理后只包含文本的测试集
│   ├── train.txt                                  // 原训练集               
│   ├── train_label.txt                         // 预处理后只包含标签的训练集        
│   └── train_text.txt                          // 预处理后只包含文本的训练集
├── output
│   └── 周恩帅-PB17111561-X.txt	  // 测试集上的若干次预测结果
├── res                                               // 一些预训练模型文件
│   ├── exp2.pdf
│   ├── glove.6B.100d.txt
│   ├── glove.6B.200d.txt
│   ├── glove.6B.300d.txt
│   ├── glove.6B.50d.txt
│   └── 实验二在线评测说明.pdf
├── src                                                 
│   ├── re-bert-lstm-conv.ipynb        // albert+lstm+cnn 模型
│   ├── re-bert-lstm.ipynb                 // albert+lstm模型
│   ├── re-bert-lstm.py                      // albert+lstm模型
│   ├── re-fasttext.ipynb                    // fasttext+lstm模型
│   ├── re-glove.ipynb                       // glove+lstm模型
│   ├── stat-feature.ipynb                  // 统计样本的一些特征，如句子长度、关系词之间的距离
│   └── util.py                                     // 预处理、数据集读取、写入函数
└── 实验报告.pdf                                
```

## 运行环境

软硬件环境：

+ Linux

+ Python3.6 with numpy，tensorflow，tensorflow_hub，tensorflow_text
+ ALBERT预训练模型
+ Nvidia Tesla V100

运行方式：

1. 预处理数据集：

   `python3 util.py`

2. 训练模型，训练完成后同时输出在测试集上的预测结果：

   `python3 re-bert-lstm.py`

## 关键函数

### 加载模型、文本预处理

从tf hub上加载预训练的ALBERT模型，将句子编码成序列化的向量

```python
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
```

### 双向LSTM分类

使用tensorflow搭建LSTM分类网络并训练

```python
x = sequence_output
x = layers.Bidirectional(layers.LSTM(lstm_nodes, recurrent_dropout=0.2, dropout=0.2))(x)
outputs = layers.Dense(len(relations), activation="softmax")(x)
model = tf.keras.Model(inputs=text_inputs, outputs=outputs)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(X, Y, epochs=epochs, batch_size=32, validation_split=valid_ratio)
```

### CNN分类

在得到最好的网络之前，我们还尝试了CNN，分为卷积、池化、合并几个步骤

```python
x = sequence_output
# x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(x)
x = layers.Reshape((128, -1, 1))(x)

x2 = layers.Conv2D(1, (2, x.shape[2]), activation='relu')(x)
x3 = layers.Conv2D(1, (3, x.shape[2]), activation='relu')(x)
x4 = layers.Conv2D(1, (4, x.shape[2]), activation='relu')(x)
x5 = layers.Conv2D(1, (5, x.shape[2]), activation='relu')(x)
x6 = layers.Conv2D(1, (6, x.shape[2]), activation='relu')(x)
# x7 = layers.Conv2D(1, (7, x.shape[2]), activation='relu')(x)
# x8 = layers.Conv2D(1, (8, x.shape[2]), activation='relu')(x)
# x16 = layers.Conv2D(1, (16, x.shape[2]), activation='relu')(x)
# x32 = layers.Conv2D(1, (32, x.shape[2]), activation='relu')(x)
# x64 = layers.Conv2D(1, (64, x.shape[2]), activation='relu')(x)

x2 = layers.MaxPooling2D(pool_size=(x2.shape[1], 1), padding='valid')(x2)
x3 = layers.MaxPooling2D(pool_size=(x3.shape[1], 1), padding='valid')(x3)
x4 = layers.MaxPooling2D(pool_size=(x4.shape[1], 1), padding='valid')(x4)
x5 = layers.MaxPooling2D(pool_size=(x5.shape[1], 1), padding='valid')(x5)
x6 = layers.MaxPooling2D(pool_size=(x6.shape[1], 1), padding='valid')(x6)
# x7 = layers.MaxPooling2D(pool_size=(x7.shape[1], 1), padding='valid')(x7)
# x8  = layers.MaxPooling2D(pool_size=(x8 .shape[1], 1), padding='valid')(x8 )
# x16 = layers.MaxPooling2D(pool_size=(x16.shape[1], 1), padding='valid')(x16)
# x32 = layers.MaxPooling2D(pool_size=(x32.shape[1], 1), padding='valid')(x32)
# x64 = layers.MaxPooling2D(pool_size=(x64.shape[1], 1), padding='valid')(x64)

x2 = layers.Reshape((-1,))(x2)
x3 = layers.Reshape((-1,))(x3)
x4 = layers.Reshape((-1,))(x4)
x5 = layers.Reshape((-1,))(x5)
x6 = layers.Reshape((-1,))(x6)
# x7 = layers.Reshape((-1,))(x7)
# x8  = layers.Reshape((-1,))(x8 )
# x16 = layers.Reshape((-1,))(x16)
# x32 = layers.Reshape((-1,))(x32)
# x64 = layers.Reshape((-1,))(x64)

# [x2, x3, x4, x5, x6, x7, x8, x16, x32, x64]
x = layers.Concatenate(axis=1)([x2, x3, x4,  x5, x6])

outputs = layers.Dense(len(relations), activation='softmax')(x)
model = tf.keras.Model(inputs=text_inputs, outputs=outputs)
```

### 加载Glove模型

在使用BERT之前，我们尝试了Glove等预训练词向量：

```python
glove_dir = os.path.join("..", "res", "")
embedding_dim = 300

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.%dd.txt' % embedding_dim), encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
 model = tf.keras.models.Sequential()

model.add(layers.Embedding(max_words, embedding_dim, mask_zero=True))

model.add(layers.Bidirectional(layers.LSTM(128, go_backwards=True,recurrent_dropout=0.2, dropout=0.2)))

model.add(layers.Dense(len(relations), activation='softmax'))

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
```