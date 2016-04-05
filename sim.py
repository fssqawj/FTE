# coding: utf8
import numpy as np

def cos(a, b):
    '''
        计算Cos相似度
    '''
    a = np.array(a)
    b = np.array(b)

    num = float(a.dot(b.T))
    denom = np.linalg.norm(a) * np.linalg.norm(b)  
    # 余弦值
    cos = 0
    if denom > abs(1e-6): 
        cos = num / denom
    # 归一化 
    sim = 0.5 + 0.5 * cos 
    return sim

if __name__ == '__main__':
    print cos([1,2,3], [-4,-5,-6]) # 0.126
    print cos([1,2,3], [-4,-5,-6]) # 0.987
