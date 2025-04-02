# kagglehab.py


import os
import kagglehub
import shutil
import csv
from lib_ts import TsLib
# from lib_ts import \
# loadSouceFile, \
# addTestData, \
# testLog



# kagglehub.login()

#### load src from kaggle.com

# local dir
src_dir_path = os.path.dirname(os.path.realpath(__file__))+'/src'
fn='/freelancer_earnings_bd.csv' # file name
pdest = src_dir_path+fn # path destination
dest_ex = os.path.isfile(pdest)


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

ptest = pdest.split('.')[0]+'_test.'+pdest.split('.')[1] # path destination
data_ex = os.path.isfile(ptest)

if not data_ex:
    tst.addTestData(pdest, ptest, 10)

data_ex = os.path.isfile(ptest)
if not dest_ex:
    exit('test source data not exists')
else:
    print('test source data is exists')


