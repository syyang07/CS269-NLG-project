# OpenNMT
This is the implementation of OpenNMT model.

#### Reference
> Klein, G. et al. “OpenNMT: Open-Source Toolkit for Neural Machine Translation.” ArXiv abs/1701.02810 (2017): n. pag. [Paper in arXiv](https://arxiv.org/abs/1701.02810).


## Introduction
In this work, we apply NMT which improves the generation process by defining a context for the each generated fake review.

## Environment Requirement
The code has been tested running under Python 3.7. The required packages are as follows:
* Pytorch==1.1.0
* torchvision==0.2.1
* tqdm==4.30.*


## How to run
Follow the instruction in each jupyter notebook. The corresponding data path is in upper level of this file, remenber to use the correct address.


## Additional information
Data wrangling with PySpark, tokenisation with CoreNLP from Stanford.

Training a Seq2Seq model with a fork of OpenNMT-py, this part is heavily inspired by Stay On-Topic: Generating Context-specific Fake Restaurant Reviews by Juuti et al.

Training various classifiers with Spark ML that try to distinguish between fake and and real reviews.


================================

