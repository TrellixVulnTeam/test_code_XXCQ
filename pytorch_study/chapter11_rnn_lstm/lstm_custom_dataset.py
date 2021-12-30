"""

This is a simple LSTM code using custom dataset called "Movie Review Sentiment Analysis".
It provides sentences of movie reviews and corresponding labels. See the dataset link for details.

- Dataset link: https://www.kaggle.com/c/movie-review-sentiment-analysis-kernels-only/data
- Code reference: https://wangjiosw.github.io/2020/02/29/deep-learning/torchtext_use/

"""

import torch
from torch import nn
from torch.nn import functional as F
from torchtext import data, datasets
from torchtext.data.utils import get_tokenizer
import pandas as pd


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test.tsv', sep='\t')

# from torchtext.datasets import IMDB
# import spacy

# Dataset Link: https://www.kaggle.com/c/movie-review-sentiment-analysis-kernels-only/data

print('GPU:', torch.cuda.is_available())
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    # spacy_en = spacy.load('en_core_web_sm')

    TEXT = data.Field(tokenize=tokenizer)
    LABEL = data.LabelField(dtype=torch.float)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    print('Len(train_data): {}, len(test_data): {}'.format(len(train_data), len(test_data)))
    

    # def tokenize(label, line):
    #     return line.split()

    # tokens = []
    # for label, line in train_iter:
    #     tokens += tokenize(label, line)


if __name__ == '__main__':
    main()