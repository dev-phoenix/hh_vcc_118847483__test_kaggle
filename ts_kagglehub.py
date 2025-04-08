# kagglehab.py


import os
import kagglehub
import shutil
import csv
from lib_ts import TsLib
import lib_model_names



# kagglehub.login()

#### load src from kaggle.com

def kaggle_test(way):
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
    tst = TsLib(src_dir_path,fn)
    tst.testFilePath = ptest
    tst.fullFilePath = pdest
    if not dest_ex: # load test data from kagglehub


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
        if not data_ex:
            exit('test source data not exists')
        else:
            print('test source data is exists')

        print('='*30)

    #
    # Test LLM Libraries
    #
    # way = 1 # select to torch version
    # way = 2 # select to langchain version
    # way = 3 # select to console choice version
    if way == 1:
        # first test llm model
        import lib_llm__torch # fail to load torch. Library torch needed 795Mb. too little memory
        if 10 : lib_llm__torch.test_llm__torch()
        # fail to loat torch. Library torch needed 795Mb. too little memory
    elif way == 2:
        # second test llm model
        import lib_llm__langch # try langchain
        lib_llm__langch.test_llm__langch()
    elif way == 3:
        print('have not satisfactory result with LLM')
        print()
        tst.dialog()
    elif way == 4:
        # experemental with huggingface
        import ts_tch_ds
        ts_tch_ds.ts_run()
    elif way == 5:
        modelLoadInfo(False)
    elif way == 6:
        modelLoadInfo(True)
    elif way == 7:
        modelLoadInfo(True, False)

def modelLoadInfo(result_info=False, all=True):
    # print('ts que gen')
    # gen = lib_model_names.smnGen(-1,True)
    # for sg in gen:
    #     print('>> 1')
    #     print(sg)
    # print('>> e')
    # print(gen.next())
    num = 0
    for model_name, test_result, llm_result_is in lib_model_names.smnGen(-1,True) :
        num+=1
        if not all and not llm_result_is: continue
        print('-'*10, num)
        print(f'LLM name: {model_name}')
        print(f'Test result: {llm_result_is}')
        if result_info:
            print(f'Test info: {test_result}')
    print('End.')
    print()

vars='''
1. select torch version
2. select langchain version (fail)
3. select console dialog version (recomended)
4. select more or less but live test version (small and live)
5. print tested libraries list
6. print tested libraries list with info
7. print tested libraries list with run success

default is 3
'''

if __name__ == '__main__':
    try:
        while True:
            maxchoice = 7
            print (vars)
            inptpl = 'Select your choice\n'
            way = input(inptpl)
            if way == '': way = 3
            print(f'You\'r choice is: {way}')
            way = int(way)
            if  way<1 or way > maxchoice:
                raise KeyboardInterrupt('Result out of range')
            kaggle_test(way)
            print('Bye.-----------------')
    except KeyboardInterrupt as e:
        print(e)
        print('Bye.')
    except ValueError as e:
        print(e)
        print('Result not numeric')
        print('Bye.')

