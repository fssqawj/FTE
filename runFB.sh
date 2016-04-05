#!/bin/bash
# python main.py --latent 10 --lmbda 0 --train data/train.txt  --test data/test.txt --log roc.log --result roc.txt
# python main.py --latent 10 --lmbda 0 --train data/wordnet-mlj12-train.txt  --test data/wordnet-mlj12-test.txt --log roc.log --result roc.txt
# python main.py --latent 10 --lmbda 0 --train data/wordnet/wordnet-mlj12-train.txt  --test data/wordnet/wordnet-mlj12-test.txt --log wordnet.log --result result_wordnet.txt

# ipython main.py -- --train  data/train.txt --test data/test.txt --log test.log --result test.log --embed exp/zzl_embedding_unstructured.pkl
# ipython main.py -- --latent 10 --lmbda 0 --train data/pad_train.txt  --test data/pad_test.txt --log test.log --result test.log --embed exp/jd_embedding.pkl
# ipython main.py -- --latent 10 --lmbda 0 --train data/foaf-train.txt  --test data/foaf-test.txt --log test.log --result test.log --embed exp/foaf_embedding.pkl
ipython myMain_FB.py -- --latent 10 --lmbda 0 --train myData/FB15k/train.txt  --test myData/FB15k/test_1000.txt --log test.log --result test.log --spath exp/FB15k_embedding_
