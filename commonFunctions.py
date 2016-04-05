from numpy import dot, loadtxt, ones
from numpy.random import randint
from numpy.random import random_integers
from scipy.sparse import csr_matrix
import numpy as np
import fnmatch
import os

def dict_link_index(l):
    '''
        Convert list to dictionary with index
        ["a", "b", "c"] to {"a":0, "b":1, "c":2}
    '''
    return {v:i for i, v in enumerate(l)}

def uniq_list(l):
    '''
        Distinct list elements
        [1,1,2] to [1,2]
    '''
    return list(set(l))

def normalize(feature=None):
    max_value = max(feature)
    min_value = min(feature)
    for i, v in enumerate(feature):
        feature[i] = 0 if max_value == min_value else (v - min_value * 1.0) / (max_value - min_value)
    return feature

def squareFrobeniusNormOfSparseBoolean(M):
    rows, cols = M.nonzero()
    return len(rows) 

def squareFrobeniusNormOfSparse(M):
    """
    Computes the square of the Frobenius norm
    """
    rows, cols = M.nonzero()
    norm = 0
    for i in range(len(rows)):
        norm += M[rows[i],cols[i]] ** 2
    return norm

def trace(M):
    """ Compute the trace of a sparse matrix
    """
    return sum(M.diagonal())

def fitNorm(X, A, R):   
    """
    Computes the squared Frobenius norm of the fitting matrix || X - A*R*A^T ||,
    where X is a sparse matrix
    """
    return squareFrobeniusNormOfSparse(X) + fitNormWithoutNormX(X, A, R)

def fitNormWithoutNormX(X, A, R):
    AtA = dot(A.T, A)
    secondTerm = dot(A.T, dot(X.dot(A), R.T))
    thirdTerm = dot(dot(AtA, R), dot(AtA, R.T))
    return np.trace(thirdTerm) - 2 * trace(secondTerm)
    

def reservoir(it, k):
    ls = [next(it) for _ in range(k)]
    for i, x in enumerate(it, k + 1):
        j = randint(0, i)
        if j < k:
            ls[j] = x
    return ls  

def checkingIndices(M, ratio = 1):
    """
    Returns the indices for computing fit values
    based on non-zero values as well as sample indices
    (the sample size is proportional to the given ratio ([0,1]) and number of matrix columns)
    """
    rowSize, colSize = M.shape
    nonzeroRows, nonzeroCols = M.nonzero()
    nonzeroIndices = [(nonzeroRows[i], nonzeroCols[i]) for i in range(len(nonzeroRows))]                
    sampledRows = random_integers(0, rowSize - 1, round(ratio*colSize))
    sampledCols = random_integers(0, colSize - 1, round(ratio*colSize))
    sampledIndices = zip(sampledRows, sampledCols)
    indices = list(set(sampledIndices + nonzeroIndices))
    return indices

def loadX(inputDir, dim):
    X = []
    numSlices = 0
    numNonzeroTensorEntries = 0
    for inputFile in os.listdir('./%s' % inputDir):
        if fnmatch.fnmatch(inputFile, '[0-9]*-rows'):
            numSlices += 1
            row = loadtxt('./%s/%s' % (inputDir, inputFile), dtype=np.uint32)
            if row.size == 1: 
                # Let row to array if its size is 1
                row = np.atleast_1d(row)
            col = loadtxt('./%s/%s' % (inputDir, inputFile.replace("rows", "cols")), dtype=np.uint32)
            if col.size == 1: 
                # Let col to array if its size is 1
                col = np.atleast_1d(col)
            Xi = csr_matrix((ones(row.size),(row,col)), shape=(dim,dim))
            # print "----------------------------"
            # print row
            # print col
            # print Xi.todense()
            # print "----------------------------"
            numNonzeroTensorEntries += row.size
            X.append(Xi)
            print 'loaded %d: %s' % (numSlices, inputFile)
    print 'The number of tensor slices: %d' % numSlices
    print 'The number of non-zero values in the tensor: %d' % numNonzeroTensorEntries
    return X
