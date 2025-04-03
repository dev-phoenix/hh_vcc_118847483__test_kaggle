# kagglehab.py


import os
import kagglehub
import shutil
import csv
from lib_ts import TsLib

import lib_llm__torch # fail to load torch. Library torch needed 795Mb. too little memory
import lib_llm__langch # try langchain


# kagglehub.login()

#### load src from kaggle.com

# local dir
src_dir_path = os.path.dirname(os.path.realpath(__file__))+'/src'
fn='/freelancer_earnings_bd.csv' # file name
pdest = src_dir_path+fn # path destination
dest_ex = os.path.isfile(pdest)

#
# Load Kaggle Data
#
ptest = pdest.split('.')[0]+'_test.'+pdest.split('.')[1] # path destination
data_ex = os.path.isfile(ptest)
if not dest_ex: # load test data from kagglehub

    tst = TsLib(src_dir_path,fn)

    if not dest_ex:
        tst.loadSouceFile()

    dest_ex = os.path.isfile(pdest)
    if not dest_ex:
        exit('source data not exists')
    else:
        print('source data is exists')

    #### copy short data slice to local

    islogged = 1
    if islogged: tst.testLog()

    if not data_ex:
        tst.addTestData(pdest, ptest, 10)

    data_ex = os.path.isfile(ptest)
    if not dest_ex:
        exit('test source data not exists')
    else:
        print('test source data is exists')

    print('='*30)

#
# Test LLM Libraries
#
way = 2 # select to langchain version
# way = 1 # select to torch version
way = 3 # select to nothing
if way == 1:
    # first test llm model
    if 10 : lib_llm__torch.test_llm__torch()
    # fail to loat torch. Library torch needed 795Mb. too little memory
elif way == 2:
    # second test llm model
    lib_llm__langch.test_llm__langch()
elif way == 3:
    print('have not sutisfactory result')