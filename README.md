## NLPer集合
### 1. 词向量和语言模型
#### Word2Vec (2013)
- 论文地址：[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- 简介：本文提出了**CBOW**模型和**Skip-Gram**模型，用来学习`word vector`。
- 代码实现：[skip-gram.ipynb](https://github.com/codecat0/NLP/blob/master/word2vec/skip_gram.ipynb)

### 2. 预训练模型
#### ELMo (2018)
- 论文地址：[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)
- 简介：`ELMo`只预训练`language model`，而`word embedding`是通过输入的句子实时输出的， 这样可以得到与上下文相关的动态`word embedding`，很大程度上缓解了歧义的发生。
- 代码实现：[elmo.py](https://github.com/codecat0/NLP/blob/master/pretrained_models/elmo/elmo.py)
#### GPT (2018)
- 论文地址：[Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)
- 简介：
#### Bert (2018)
- 论文地址：[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- 简介

### 3. 文本分类
#### 3.1 中文情感分析所用的模型
##### TextCNN (2014)
- 论文地址：[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882)
- 简介：利用多种卷积核大小提取序列的局部区域特征
- 代码实现：[TextCNN.py](https://github.com/codecat0/NLP/blob/master/text_classification/emotion_cls/models/TextCNN.py)
##### TextRNN (2015)
- 论文地址：[Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)
- 简介：采用双向LSTM结构，更好地捕获句子中的语义特征
- 代码实现：[TextRNN.py](https://github.com/codecat0/NLP/blob/master/text_classification/emotion_cls/models/TextRNN.py)
##### Text Bi-LSTM Attention (2016)
- 论文地址：[Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](https://aclanthology.org/P16-2034.pdf)
- 简介： 在双向LSTM结构之上加入Attention机制，结合上下文更好地表征句子语义特征
- 代码实现：[TextBiLSTM_Att.py](https://github.com/codecat0/NLP/blob/master/text_classification/emotion_cls/models/TextBiLSTM_Att.py)

##### Text Bi-LSTM Hierarchical Attention (2016)
- [Hierarchical Attention Networks for Document Classiﬁcation](https://aclanthology.org/N16-1174.pdf)
- 简介：在双向LSTM结构之上加入Hierarchical Attention机制，结合上下文更好地表征句子语义特征
- 代码实现：[TextBiLSTM_Att.py](https://github.com/codecat0/NLP/blob/master/text_classification/emotion_cls/models/TextBiLSTM_Att.py)