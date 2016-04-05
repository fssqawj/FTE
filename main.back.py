# coding: utf8
import numpy as np
import cPickle as pickle
import sys
import time
from gd import GD
from logger import Logger
from commonFunctions import *
from sim import cos
from datetime import datetime
from numpy.linalg import norm 
from dataset import dataset
from rescal import *
from eval import *
from util import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

class RunRescal:
    '''
        封装了测试集、训练集、RESCAL算法、logger
    '''
    def __init__(self, train, test, logger):
        '''
            train:  训练集
            test:   测试集
            logger: logger
        '''
        self.train = train
        self.test = test
        self.logger = logger
        self.train.load_triples()
        self.test.load_triples()
        self.X = self.train.build_csr_matrix()
        self.A = None
        self.R = None
        self.scores = None

    def rescal(self, config, dump=True):
        start_time = datetime.now()
        if dump:
            factorization = rescal(self.X, config['numLatentComponents'], lmbda=config['regularizationParam'])
            self.A = factorization[0]
            self.R = factorization[1]
            pickle.dump(self.A, open('exp/rescalA.pkl', 'w'))
            pickle.dump(self.R, open('exp/rescalR.pkl', 'w'))
            self.logger.getLog().info("Finished dump rescal experiment")
        else:
            self.A = pickle.load(open('exp/rescalA.pkl', 'r'))
            self.R = pickle.load(open('exp/rescalR.pkl', 'r'))
        end_time = datetime.now()
        self.logger.getLog().info("Finished RESCAL in %ds" % ((end_time - start_time).seconds))

    def tranE(self, embeddings):
        start_time = datetime.now()
        self.E = embeddings
        end_time = datetime.now()
        self.logger.getLog().info("Finished TranE in %ds" % ((end_time - start_time).seconds))

    def calEntityPairScoreRescal(self, enIdx1, enIdx2): 
        '''
        Return RESCAL score:
            [('father': 0.9), ('mother': 0.5), ...]
        '''
        # scores = [(self.train.get_all_relation()[i], np.dot(self.A[enIdx1,:], np.dot(self.R[i], self.A.T[:, enIdx2]))) for i in range(len(self.R))]
        scores = [np.dot(self.A[enIdx1,:], np.dot(self.R[i], self.A.T[:, enIdx2])) for i in range(len(self.R))]
        return scores

    def calEntityPairScoreTransE(self, enIdx1, enIdx2):
        '''
        Return TransE score:
            [('father': 0.9), ('mother': 0.5), ...]
        '''
        e1 = self.train.get_all_entity()[enIdx1]
        e2 = self.train.get_all_entity()[enIdx2]
        all_relation = self.train.get_all_relation()
        # scores = [(all_relation[i], cos(self.E[all_relation[i]], minusVector(self.E[e2], self.E[e1]))) for i in range(len(self.R))]
        scores = [cos(self.E[all_relation[i]], minusVector(self.E[e2], self.E[e1])) for i in range(len(self.R))]
        return scores

    def makeX(self, enIdx1, enIdx2):
        '''
            对一个实体对产生梯度下降所需要的x值(x1, x2)，x1是transE值，x2是RESCAL值
            Return: [[0.1, 0.4], [0.5, 0.2], ...] length: 关系的个数
        '''
        transEScore = normalize(self.calEntityPairScoreTransE(enIdx1, enIdx2))
        rescalScore = normalize(self.calEntityPairScoreRescal(enIdx1, enIdx2))
        return np.array([[transEScore[i], rescalScore[i]] for i in range(len(transEScore))])

    def makeY(self, enIdx1, enIdx2, r):
        '''
            对一个实体对产生梯度下降所需要的y值，1为实体1和实体2在测试集中的关系，其他的均为0
            Return: [1, 0, 0, 0, 0 ] length: 关系的个数
        '''
        relations = self.train.get_all_relation()
        y = [1 if r == relations[i] else 0 for i in range(len(relations))]
        return np.array(y)

    def startDescentGradient(self, idx_e1, idx_e2, r, er):
        x = self.makeX(idx_e1, idx_e2)
        y = self.makeY(idx_e1, idx_e2, r)
        gd = GD(er, x, y)
        result = gd.start()
        return result

    def calEveryRelationScore(self, t1, t2):
        '''
        Return scores:
            {
                1:[('father': 0.9), ('mother': 0.5), ...],
                2:[('son': 0.8), ('brother': 0.2), ...],
                3:[],  因为3没有出现在训练集中,所以没有score
                ...
                # 列表没有排序
            }
        '''
        self.logger.getLog().info("Starting calEveryRelationScore")
        scores = {}
        for idx, triple in enumerate(self.test.get_triple()):
            e1, r, e2 = triple[0:3]
            try:
                idx_e1 = self.train.get_idx_entity()[e1]
                idx_e2 = self.train.get_idx_entity()[e2]
            except:
                # self.logger.getLog().error("在训练集中没有找到实体或关系: %" % (','.join(triple)))
                self.logger.getLog().error("在训练集中没有找到实体或关系: " + str(triple))
                scores[idx] = []
                continue
            # scores[idx] = self.calEntityPairScoreRescal(idx_e1, idx_e2)
            transEScore = normalize(self.calEntityPairScoreTransE(idx_e1, idx_e2))
            rescalScore = normalize(self.calEntityPairScoreRescal(idx_e1, idx_e2))
            scores[idx] = []
            for i in range(len(transEScore)):
                scores[idx].append((self.train.get_all_relation()[i], t1*transEScore[i] + t2*rescalScore[i]))
        self.scores = scores
        return scores

    def training(self, er):
        '''
        Return thetas: 
            t1: RESCAL算法的参数
            t2: transE算法的参数
        '''
        self.logger.getLog().info("Starting training")
        scores = {}
        thetas = []
        for idx, triple in enumerate(self.test.get_triple()):
            e1, r, e2 = triple[0:3]
            try:
                idx_e1 = self.train.get_idx_entity()[e1]
                idx_e2 = self.train.get_idx_entity()[e2]
            except:
                self.logger.getLog().error("在训练集中没有找到实体或关系: " + str(triple))
                scores[idx] = []
                continue
            # 梯度下降学习参数
            thetas.append(self.startDescentGradient(idx_e1, idx_e2, r, er))
        thetas = np.array(thetas)
        # print np.mean(thetas[:, 1, 0])
        # print np.mean(thetas[:, 0, 0])
        t1 = np.mean(thetas[:, 0, 0])
        t2 = np.mean(thetas[:, 1, 0])
        self.logger.getLog().info("t1: %d, t2:%d" % (t1, t2))
        return t1, t2

    def pickPredictedResult(self, threshold=0):
        '''
        在测试集的三元组上加上预测的关系，(en1, rel, en2) -> (en1, rel, en2, rel_predicted)
        Return testCase:
            [['en1', 'rel1', 'en2', 'prerel1'], ...]
        '''
        logger.getLog().info("Starting pickPredictedResult")
        testCase = self.test.get_triple()
        for idx, scoreHash in self.scores.items():
            scoreHash = sorted(scoreHash, key=lambda x: x[1], reverse=True)
            if scoreHash[0][1] >= threshold:
                # [('father', 0.9), ...]  取出得分最高的关系名
                testCase[idx].append(scoreHash[0][0])

        # for idx in testCase:
        #     print ','.join(idx)

        return testCase

    # ROC, AUC, PRECISION, RECALL
    def evaluateEntity(self, method, config, verbose=False):
        pass

    def evaluateRelation(self, method, config, verbose=False):
        pass

    def evaluate_relation_ROC(self, threshold=0):
        # Result: "[[relation, true_relation], [relation, true_relation], ...]"
        result = []
        total = self.test.get_num_triple()
        for triple in self.test.get_triple():
            e1, r, e2 = triple[0:3]
            idx_e1 = self.train.get_idx_entity()[e1]
            idx_e2 = self.train.get_idx_entity()[e2]
            actual_relation = self.train.get_idx_relation()[r.strip()]
            scores = [np.dot(self.A[idx_e1,:], np.dot(self.R[i], self.A.T[:, idx_e2])) for i in range(len(self.R))]
            for i, score in enumerate(scores):
                self.logger.getLog().debug("Score: %s\t%s\t%s: %f" % (e1, self.train.get_all_relation()[i], e2.strip(), score))
            maxIdx = maxIndex(scores)
            result.append([r, self.train.get_all_relation()[maxIdx]])
        return result

    def roc(self, evaluation):
        # for e in evaluation:
        #     print ','.join(e)
        allRel = []
        totalScore = 0
        totalAccuracy = 0
        for e in evaluation:
            allRel.append(e[0])
            allRel.append(e[1])
        allRel = list(set(allRel))

        for i, rel in enumerate(allRel):
            true = []
            score = []
            for sample in evaluation:
                true.append(1 if sample[0] == rel else 0)
                score.append(1 if sample[1] == rel else 0)
            try:
                totalScore += roc_auc_score(np.array(true), np.array(score))
                totalAccuracy += accuracy_score(np.array(true), np.array(score))
            except:
                # print rel
                # print score
                continue

        count = 0
        for sample in evaluation:
            if sample[0] == sample[1]:
                count += 1
        self.logger.getLog().info("Çount: %d, Evaluation length: %d" % (count, len(evaluation)))
        # print "Count:", count, " Evaluation: ", len(evaluation)
        return totalScore * 1.0 / len(allRel), totalAccuracy * 1.0 / len(allRel)

if __name__ == '__main__':
    start_time = datetime.now()
    dataArgs, algoArgs = parseArguments()

    logFile = "./log/" + dataArgs['log']
    resultToFile = open("./result/" + dataArgs['result'], 'w')

    # 初始化logger和算法实例
    logger = Logger()
    runRescal = RunRescal(dataset(dataArgs['train'], "UTF-8"), dataset(dataArgs['test'], "UTF-8"), logger)

    # 运行RESCAL和Tranlating Embedding算法
    runRescal.rescal(algoArgs, True)
    runRescal.tranE(loadPickle(dataArgs['embed']))
    t1, t2 = runRescal.training(0.001)
    runRescal.calEveryRelationScore(t1, t2)
    testCase = runRescal.pickPredictedResult()
    roc, acc = runRescal.roc([(i[-1], i[-3]) for i in testCase])
    print "ROC: %f, ACC: %f" % (roc, acc)
    end_time = datetime.now()
    logger.getLog().info("Totally finished in %ds" % ((end_time - start_time).seconds))