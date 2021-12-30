"""
Sample LSTM-based sentence label classification with custom dataset

This is a simple LSTM code using custom dataset called "Movie Review Sentiment Analysis".
It provides sentences of movie reviews and corresponding labels. See the dataset link for details.

- Dataset link: https://www.kaggle.com/c/movie-review-sentiment-analysis-kernels-only/data
- Code reference: https://wangjiosw.github.io/2020/02/29/deep-learning/torchtext_use/

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torchtext
from torchtext import data, datasets
from torchtext.data.utils import get_tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy

import argparse
import os
from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser(description='Test_setting')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epoch number')
    parser.add_argument('--tensorboard_window_name', type=str,
                        default='')
    parser.add_argument('--save_models_path', type=str, default='')
    args = parser.parse_args()

    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    # ## Parse raw dataset and split training and validation sets
    #
    # Dataset link: https://www.kaggle.com/c/movie-review-sentiment-analysis-kernels-only/data.
    #
    # This dataset contains some tsv files (actually the same format of .csv files).
    # I have already downloaded this dataset and put in `data` folder.

    # Read csv files first
    raw_train_data = pd.read_csv('./data/train.tsv', sep='\t')
    raw_test_data = pd.read_csv('./data/test.tsv', sep='\t')

    # Print some samples
    print('data.shape: {}, test.shape: {}'.format(
        raw_train_data.shape, raw_test_data.shape))
    print(raw_train_data[:5])
    print(raw_test_data[:5])

    # Create train and validation set (both from training dataset)
    train_part, val_part = train_test_split(raw_train_data, test_size=0.2)
    # Need to save csv files. Will be loaded later to create data iterator.
    train_part.to_csv("./data/train_split.csv", index=False)
    val_part.to_csv("./data/val_split.csv", index=False)

    # Build vocabulary
    #
    # Steps:
    # 1. Create Field types TEXT and LABEL.
    # 2. Create Dataset objects for training, testing, validation;
    # 3. Build vocabulary.

    # Create tokenizer. Two ways, both work:
    # 1) Define a function
    spacy_en = spacy.load('en_core_web_sm')

    def tokenizer(text):  # create a tokenizer function
        return [tok.text for tok in spacy_en.tokenizer(text)]

    # 2) Use torchtext API.
    tokenizer_api = get_tokenizer('spacy', language='en_core_web_sm')

    # Field
    TEXT = data.Field(sequential=True, tokenize=tokenizer_api, lower=True)
    LABEL = data.Field(sequential=False, use_vocab=False)

    # Dataset objects. Here we use API 'data.TabularDataset' which defines a dataset for CSV,
    # TSV or json format.
    # Official Ref: https://torchtext.readthedocs.io/en/latest/data.html#tabulardataset
    # NOTE for parameters:
    # - If your csv file has a header row, remember to set 'skip_header=True'.
    # - In 'fields', the order must be exactly the same as order in csv files.
    # - You can also define your own Dataset class instead of using Torchtext API. See
    #   a ref here: https://blog.nowcoder.net/n/3a8d2c1b05354f3b942edfd4966bb0c1.

    train, val = data.TabularDataset.splits(
        path='.', train='./data/train_split.csv', validation='./data/val_split.csv',
        format='csv', skip_header=True,
        fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT), ('Sentiment', LABEL)])

    test = data.TabularDataset(path='./data/test.tsv', format='tsv', skip_header=True,
                               fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT)])

    # Build vocabulary from words in input data. This will also save the indices of words inside
    # the vocabulary (stored inside 'train' object). NOTE that words in vocabulary will be sorted
    # by word frequency.
    # - 'glove.6B.100d' is one encoding method provided by torchtext. Here 100d means each word is
    #   encoded as a 100-vector. This is like some word-to-encoding dictionary and will be
    #   downloaded to cache for the first time (about 900M).
    # - 'max_size' is to constrain the total number of encoded words. Other words not in this
    #   dictionary will be encoded to some default initialized vectors. This is reasonable to save
    #   space and time, since common words are not too many. If this parameter is not used,
    #   it will create vocabulary for all words met in the training data.
    #   NOTE that if this parameter is provided (such as 30000), and identical words is > 30000,
    #   then vocabulary's size will be actually 30002 instead of 30000, with 1 index for unmet
    #   words and another 1 for nothing (not very sure).
    TEXT.build_vocab(train, vectors='glove.6B.100d', max_size=30000)

    # This is to provide some initialization way for words not in dictionary.
    # 'xavier_uniform' is some method in a 2010 paper.
    # Official Ref: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
    TEXT.vocab.vectors.unk_init = nn.init.xavier_uniform
    len_vocab = len(TEXT.vocab)
    print('len_vocab:', len_vocab)
    print(type(test.examples))
    print(test.examples[0].Phrase)

    flag_tb_viewer = False
    if args.tensorboard_window_name:
        flag_tb_viewer = True
        writer = SummaryWriter('runs/' + args.tensorboard_window_name)

    # Create iterators
    #
    # An iterator is to:
    # - 将 Datasets 中的数据 batch 化
    # - 其中会包含一些 pad 操作，保证一个 batch 中的 example长度一致
    # - 在这里将 string token 转化成 index
    #
    # Torchtext provides many iterator types. Here we use BucketIterator which:
    # - Defines an iterator that batches examples of similar lengths together.
    # - Minimizes amount of padding needed while producing freshly shuffled batches for each new epoch.
    #
    # BucketIterator：相比于标准迭代器，会将类似长度的样本当做一批来处理，因为在文本处理中经常会需要将每一批样本长度
    # 补齐为当前批中最长序列的长度。因此当样本长度差别较大时，使用BucketIerator可以带来填充效率的提高。
    #
    # 除此之外，我们还可以在Field中通过fix_length参数来对样本进行截断补齐操作。
    #

    # Iterator.
    # 1.将 Datasets 中的数据 batch 化
    # 2.其中会包含一些 pad 操作，保证一个 batch 中的 example长度一致
    # 3.在这里将 string token 转化成 index，即 numericalization 过程 (应该是利用了此前建的 vocabulary，
    #   不过这里并没有 vocab 相关的输入参数。个人猜想是 'train' 内已经存储了每个单词在 vocabulary 中的 index)
    batch_size = 128
    train_iter = data.BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.Phrase),
                                     shuffle=True, device=DEVICE)

    val_iter = data.BucketIterator(val, batch_size=batch_size, sort_key=lambda x: len(x.Phrase),
                                   shuffle=True, device=DEVICE)

    # NOTE: 在 test_iter, sort一定要设置成 False, 要不然会被 torchtext 搞乱样本顺序。
    test_iter = data.Iterator(dataset=test, batch_size=batch_size, train=False,
                              sort=False, device=DEVICE)

    # print(type(train.examples))
    print(val.examples[0].Phrase)

    # Print some sample
    first_batch = next(iter(train_iter))
    first_batch_data = first_batch.Phrase
    first_batch_target = first_batch.Sentiment
    print(first_batch_data.shape)
    print(first_batch_target.shape)
    print(first_batch_target)
    # The data should be integer indices of words, instead of original words.
    print(first_batch_data)

    # Network Definition
    #
    # Here we define a simple LSTM-based network.

    class MyLSTM(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, num_layers):
            super(MyLSTM, self).__init__()

            # Create this 'lookup' table. NOTE that this is only initialization. You need to
            # copy the vocabulary data to this variable outside this network class explicitly.
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=embedding_dim)

            self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                                num_layers=num_layers, bidirectional=True, dropout=0.5)

            # Here '* 2' because we are using bidirctional LSTM
            self.linear = nn.Linear(hidden_dim * num_layers * 2, num_labels)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            """
            x: [seq_len, b]
            """
            # x: [seq_len, b] => embedding [seq_len, b, embedding_dim]
            # nn.Embedding(x) is basically a lookup table, so for each element in x,
            # it will replace this element with the embedding vector found in the table.
            # This is exactly equal to adding a new dimension in the rightmost position of its size.
            # That is,  [seq_len, b] => [seq_len, b, embedding_dim], while the latter is
            # exactly the default input size of a LSTM network.
            # NOTE:
            # - This is interesting: Dropout can work on a common tensor (this is not like an unknown
            # variable like weights). This is doable.
            embedding = self.dropout(self.embedding(x))

            # out: [seq_len, b, hidden_dim]
            # h: [num_layers * 2, b, hidden_dim] (here multiplying 2 because we are using bidirctional LSTM)
            # c: [num_layers * 2, b, hidden_dim]
            net_out, (h, c) = self.lstm(embedding)
            # print('net_out: {}, h: {}, c: {}'.format(net_out.shape, h.shape, c.shape))

            # [num_layers * 2, b, hidden_dim] => a tuple of [b, hidden_dim] with size 'num_layers * 2'
            h_split = h.split(1, dim=0)  # tuple of [1, b, hidden_dim]
            h_split = [x.squeeze(0)
                       for x in h_split]  # tuple of [b, hidden_dim]
            # print(h_split[0].shape)

            # concatenate this list of [b, hidden_dim] => [b, hidden_dim * num_layers * 2]
            h_cat = torch.cat(h_split, dim=1)
            # print(h_cat.shape)

            # [b, hidden_dim * num_layers * 2] => [b, num_labels]
            out = self.linear(self.dropout(h_cat))
            return out

    # Training preparation
    #
    # Here we create network-related variables, and copy vocabulary lookup table weights
    # into network.

    # There are 5 labels in this dataset.
    net = MyLSTM(vocab_size=len_vocab, embedding_dim=100,
                 hidden_dim=256, num_labels=5, num_layers=3)

    # NOTE: copy vocabulary lookup table to embedding in the network.
    vocab_vectors = TEXT.vocab.vectors
    print('vocab_vectors:', vocab_vectors.shape)
    net.embedding.weight.data.copy_(vocab_vectors)
    print('Embedding layer inited.')

    net.to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    criteon = nn.CrossEntropyLoss().to(DEVICE)
    net.to(DEVICE)

    # ## Training and testing loop
    epochs = args.epochs
    batch_num = len(list(train_iter))
    sentence_num = len(train)
    for epoch in range(epochs):

        net.train()
        for batch_idx, batch in enumerate(train_iter):
            source = batch.Phrase
            target = batch.Sentiment
            logits = net(source)
            # print(logits.shape)

            loss = criteon(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                curr_sentence_num = batch_idx * batch_size
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, curr_sentence_num, sentence_num,
                    100. * curr_sentence_num / sentence_num, loss.item()))

                if flag_tb_viewer:
                    # ...log the running loss
                    writer.add_scalar('train/loss/batch', loss.item(),
                                      epoch * sentence_num + curr_sentence_num)

        train_loss = loss.item()

        net.eval()
        with torch.no_grad():
            test_loss = 0
            correct = 0
            for batch in val_iter:
                source = batch.Phrase
                target = batch.Sentiment
                logits = net(source)
                # print(logits.shape)

                test_loss += criteon(logits, target).item()

                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum()

            test_loss /= len(val)
            accuracy = 100. * correct / len(val)
            print(
                '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(val), accuracy))

        if flag_tb_viewer:
            # Add losses and accuracy per epoch in seperate curves
            writer.add_scalar('train/loss/epoch', train_loss, epoch)
            writer.add_scalar('test/loss/epoch', test_loss, epoch)
            writer.add_scalar('test/accuracy', accuracy, epoch)

        if args.save_models_path:
            filename = 'lstm-epoch-{}-testloss-{:.2f}-accuracy-{:.2f}.mdl'.format(
                epoch, test_loss, accuracy)
            save_path = os.path.join(args.save_models_path, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
