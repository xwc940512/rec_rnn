# Contextual LSTM
I'm planning to build a Recommender System based on a contextual [LSTM](https://arxiv.org/pdf/1409.2329.pdf) network. 

## Current state

* Trying to predict a sequence of consumed movies based on [Movielens](http://grouplens.org/datasets/movielens/) data set (probably not the best use case though)
* Naive extension of LSTM implementation provided by Tensorflow
* State of gate cells @t is based on input@t, h@(t-1) and context@t
* For now, context is just the genre vector w.r.t. the sequence of input movies (somewhat ill-defined) 
* Trying to integrate benefits of [Matrix Factorization](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) into the model (for now, simply as another context variable)

First time working with Python & [Tensorflow](http://www.tensorflow.org/).
