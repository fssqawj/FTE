import numpy as np
import cPickle as pickle
from scipy.sparse import csr_matrix
from commonFunctions import dict_link_index, uniq_list


class dataset(object):
    def __init__(self, file_path, file_encode):
        self.T = None
        self.file_path = file_path
        self.file_encode = file_encode
        self._all_entity = []
        self._all_relation = []
        self._triple = []
        self._idx_relation = {}
        self._idx_entity = {}
        self._num_entity = 0
        self._num_relation = 0
        self._num_triple = 0
        # slef._ent = []
        # self._rel = []

    def __repr__(self):
        return "< Entities:%d, Relations:%d, Data Lines:%d >" % (self._num_entity, self._num_relation, self._num_triple)

    def get_all_entity(self):
        return self._all_entity

    def get_all_relation(self):
        return self._all_relation

    def get_num_entity(self):
        return self._num_entity

    def get_num_relation(self):
        return self._num_relation

    def get_num_triple(self):
        return self._num_triple

    def get_triple(self):
        return self._triple

    def get_idx_entity(self):
        return self._idx_entity

    def get_idx_relation(self):
        return self._idx_relation

    def load_triples(self, sep="\t"):
        """
            Read file contains triples like: john has_mother mary
        """
        for line in open(self.file_path, self.file_encode).read().splitlines():
            # line = line.replace(' ', '')
            e1, r, e2 = line.split(sep)
            # e1 = e1.replace(' ', '')
            # e2 = e2.replace(' ', '')
            # r = r.replace(' ', '')
            self._all_relation.append(r)
            self._all_entity.append(e1)
            self._all_entity.append(e2)
            self._triple.append([e1, r, e2])

        self._all_entity = uniq_list(self._all_entity)
        self._all_relation = uniq_list(self._all_relation)
        self._num_entity = len(self._all_entity)
        self._num_relation = len(self._all_relation)
        self._num_triple = len(self._triple)
        self._idx_relation = dict_link_index(self._all_relation)
        self._idx_entity = dict_link_index(self._all_entity)

        # return self._triple, self._all_relation, self._all_entity

    def build_matrix(self):
        T = np.zeros((self._num_entity, self._num_entity, self._num_relation))

        for e1, e2, r in self._triple:
            idx_e1 = self._idx_entity[e1]
            idx_e2 = self._idx_entity[e2]
            idx_rel = self._idx_relation[r]

            T[idx_e1, idx_e2, idx_rel] = 1
        self.T = T
        return T

    def build_csr_matrix(self):
        T = []
        row = [[] for i in range(self._num_relation)]
        col = [[] for i in range(self._num_relation)]

        for e1, r, e2 in self._triple:
            idx_e1 = self._idx_entity[e1]
            idx_e2 = self._idx_entity[e2]
            idx_rel = self._idx_relation[r]

            row[idx_rel].append(idx_e1)
            col[idx_rel].append(idx_e2)

        for i in range(self._num_relation):
            T.append(csr_matrix((np.ones(len(row[i])), (row[i], col[i])), shape=(self._num_entity, self._num_entity)))
        return T
