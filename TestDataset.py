#!/usr/bin/env python  
#encoding: utf-8  
import unittest  
from dataset import dataset

class mytest(unittest.TestCase):  
    ##初始化工作  
    def setUp(self):  
        self.data = dataset('testcase.txt', 'UTF-8')

    #退出清理工作  
    def tearDown(self):  
        pass  

    def testloadTriples(self):  
        self.data.load_triples(sep="\t")
        self.assertEqual(self.data.get_num_entity(), 36)
        self.assertEqual(self.data.get_num_relation(), 10)
        self.assertEqual(self.data.get_num_triple(), 30)

if __name__ =='__main__':  
    unittest.main()  
