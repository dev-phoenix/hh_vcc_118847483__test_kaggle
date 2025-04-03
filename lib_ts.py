# lib_ts

import os
import kagglehub
import shutil
import csv
from pathlib import Path


class TsLib:
    islogged = False
    def __init__(self, src_dir_path = '', fn=''):
        '''
        src_dir_path - path copy to
        fn - filename copy to
        '''
        if not fn:
            fn='/freelancer_earnings_bd.csv' # file name
        if not src_dir_path:
            src_dir_path = os.path.dirname(os.path.realpath(__file__))+'/src'
        self.src_dir_path = src_dir_path
        self.fn = fn


    def loadSouceFile(self):

        if not os.path.exists('src'):
            print('Dir \'src\' not exists. Creating ...')
            # os.makedirs('./src')
            Path('src').mkdir()
            print('Created.')
            
        # KAGGLE_CONFIG_DIR='./.kaggle'
        # Download latest version
        toload = "shohinurpervezshohan/freelancer-earnings-and-job-trends"
        path = kagglehub.dataset_download(toload)#, path=src_dir_path+'/feajf.csv')

        print("Path to dataset files:", path)
        # p = 'Path to dataset files: /home/eagle/.cache/kagglehub/datasets/shohinurpervezshohan/freelancer-earnings-and-job-trends/versions/1'

        pfrom = path+self.fn # path from
        pdest = self.src_dir_path+self.fn # path destination

        from_ex = os.path.isfile(pfrom)
        dest_ex = os.path.isfile(pdest)
        # print(from_ex)
        # print(dest_ex)

        if from_ex and not dest_ex:
            shutil.copy(pfrom, pdest)



    def addTestData(self,pdest,dest,cou=10):
        '''
        pdest - get from
        dest - put to
        cou - get count
        '''
        with open(pdest, newline='') as f:
            reader = csv.reader(f)
            c=cou 
            with open(dest, 'w', newline='') as f:
                writer = csv.writer(f)
                for row in reader:
                    if c<0: break
                    c-=1
                    if self.islogged: print(row)
                    writer.writerow(row)


    def testLog(self, *args, **kwargs):
        print('='*30)
        # dirs = [print (d) for d in dir(kagglehub)]
        # dirs2 = [print (d) for d in dir(kagglehub)[9:]]
        # print('='*30)


        # help(kagglehub)
        # print(kagglehub.dataset_download.__doc__)
        # help(kagglehub.dataset_download)
        # print(__file__)
        # print(os.path.realpath(__file__))
        # print(src_dir_path)
        # print('='*30)