# coding: utf8
import argparse
import cPickle

def parseDataArgs(args):
    '''
        抽取有关数据的输入参数:
        --train:   训练集路径
        --test:    测试集路径
        --embed:   Embedding向量路径
        --log:     logger文件路径
        --result:  保存结果文件路径
    '''
    return {
            'train': args.train,
            'test': args.test,
            'embed': args.embed,
            'log': args.log,
            'result': args.result
          }

def parseAlgoArgs(args):
    '''
        抽取有关算法的输入参数:
        --latent:  RESCAL算法参数
        --lmbda:   RESCAL算法参数
        --th:      RESCAL算法参数
    '''
    return {
            'numLatentComponents': args.latent,
            'regularizationParam': args.lmbda,
            'th': args.th,
          	'spath': args.spath
		   }

def parseArguments():
    '''
        接受用户传入的参数：
        --train:   训练集路径
        --test:    测试集路径
        --embed:   Embedding向量路径
        --log:     logger文件路径
        --result:  保存结果文件路径
        --latent:  RESCAL算法参数
        --lmbda:   RESCAL算法参数
        --th:      RESCAL算法参数
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="train file", required=True)
    parser.add_argument("--embed", type=str, help="Embedding pkl file", required=False)
    parser.add_argument("--test", type=str, help="test file", required=True)
    parser.add_argument("--log", type=str, help="log file", default="rescal.log", required=False)
    parser.add_argument("--result", type=str, help="result file", default="result.txt",  required=False)
    parser.add_argument("--latent", type=int, help="number of latent components", default=2, required=False)
    parser.add_argument("--lmbda", type=float, help="regularization parameter", default=0, required=False)
    parser.add_argument("--th", type=float, help="regularization parameter", default=0, required=False)
    parser.add_argument("--spath", type=str, help="save pkl file path", default="default.pkl", required=True)
    
    args = parser.parse_args()

    dataArgs = parseDataArgs(args)
    algoArgs = parseAlgoArgs(args)
    return dataArgs, algoArgs

def maxIndex(array):
    '''
        获取一个list中最大值的第一次出现的下标
    '''
    return array.index(max(array))

def loadPickle(filePath):
    '''
        序列化输出到文件
    '''
    return cPickle.load(open(filePath, 'r'))

def dumpPickle(data, filePath):
    '''
        序列化读取文件
    '''
    return cPickle.dump(data, open(filePath, 'wb'))

def minusVector(a, b):
    if len(a) != len(b) or len(a) == 0 or len(b) == 0:
        return []
    return [a[i] - b[i] for i in range(len(a))]

def plusVector(a, b):
    if len(a) != len(b) or len(a) == 0 or len(b) == 0:
        return []
    return [a[i] + b[i] for i in range(len(a))]
