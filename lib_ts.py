# lib_ts

import os
import kagglehub
import shutil
import csv
from pathlib import Path


class TsLib:
    fullFilePath = None
    testFilePath = None
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
        self.src_dir_path = src_dir_path+fn


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
    
    lowexpertCou = 0
    expertCou=0
    regions = {}
    paymethod = {}
    erningAvg = 0
    dt = None
    def loadData(self):
        fn = self.testFilePath
        fn = self.fullFilePath
        # print('fn', fn)
        if not os.path.isfile(fn): return False
        with open(fn) as fl: # fl - file
            flcsv = csv.DictReader(fl) # flcsv - file csv
            self.cou=0
            self.lowexpertCou = 0
            self.expertCou=0
            self.regions = {}
            self.paymethod = {}
            for row in flcsv:
                try:
                    # print( type(row))
                    # print(row)
                    # print('a',row['Payment_Method'])
                    # print('b', row['Payment_Method'] not in self.paymethod)
                    self.cou+=1
                    # if not self.cou: continue
                    if row['Experience_Level'] == 'Expert':
                        self.expertCou+=1
                        if int(row['Job_Completed']) <100:
                            self.lowexpertCou+=1

                    if row['Client_Region'] not in self.regions:
                        self.regions[row['Client_Region']] = {}

                    # print('c', row['Payment_Method'] not in self.paymethod)
                    if row['Payment_Method'] not in self.paymethod:
                        # print('create ', row['Payment_Method'])
                        self.paymethod[row['Payment_Method']] = {'cou':0,'sum':0,'avg':0}

                    self.paymethod[row['Payment_Method']]['cou'] +=1
                    self.paymethod[row['Payment_Method']]['sum'] +=int(row['Earnings_USD'])

                    self.paymethod[row['Payment_Method']]['avg'] = \
                    self.paymethod[row['Payment_Method']]['sum'] / \
                    self.paymethod[row['Payment_Method']]['cou']
                except Exception as ex:
                    print('err',ex)
            # print(self.paymethod)
            self.erningAvg = 0
            for ern in self.paymethod:
                self.erningAvg += self.paymethod[ern]['avg']
            self.erningAvg = self.erningAvg / len(self.paymethod)
            self.cryptOver = self.paymethod['Crypto']['avg'] - self.erningAvg
# {'Freelancer_ID': '10', 'Job_Category': 'Data Entry', 'Platform': 'PeoplePerHour', 
#  'Experience_Level': 'Beginner', 'Client_Region': 'Middle East', 
#  'Payment_Method': 'Mobile Banking', 'Job_Completed': '156', 
#  'Earnings_USD': '6608', 'Hourly_Rate': '54.99', 'Job_Success_Rate': '85.4', 
#  'Client_Rating': '4.57', 'Job_Duration_Days': '52', 'Project_Type': 'Hourly', 
#  'Rehire_Rate': '32.76', 'Marketing_Spend': '160'}


    q = None
    def dialog(self,qnum=False):
        if not self.q:
            self.q = Questions()
            self.loadData()
        if qnum == '0':    exit("\n\nСценарий завершён")
        if not qnum:
            with Questions() as qs:
                print()
                print('='*10)
                for n,q in qs.questions():
                    print(str(n)+':',q)
            mes=f"Выберите вопрос [{self.q.min} - {self.q.max}]:\n"
            try:
                self.dialog(input(mes))
            except KeyboardInterrupt:
                exit("\n\nСценарий завершён.")
            except EOFError:
                exit("\n\nСценарий завершён..")
            except Exception as e :
                print(e)
                exit("\n\nСценарий завершён...")
            exit
        # print(type(qnum))
        try :
            qnum = int(qnum)
        except:
            self.dialog()
        if int(qnum) < self.q.min or int(qnum) > self.q.max:
            # mes=f"Выберите вопрос [{self.q.min} - {self.q.max}]:\n"
            # print(mes)
            # print(qnum)
            self.dialog()
            return
        method = f'getRes_{qnum}'
        if hasattr(self, method) and callable(getattr(self, method)):
            print()
            print('>'*5)
            getattr(self, method)()
        self.dialog()
        return


    def getRes_1(self):
        # print('you chose:', 1)
        Crypto = self.paymethod['Crypto']['avg']
        print(f'Средний доход всех фрилансеров: {self.erningAvg:.2f}')
        print(f'Средний доход у фрилансеров, принимающих оплату в криптовалюте: {Crypto:.2f}')
        # over = 
        print(f"Средний доход у фрилансеров, принимающих оплату в криптовалюте, \n\
    выше дохода остальных фрилансеров, в среднем на: {self.cryptOver:.2f}")
        # print(f'{self.cryptOver:.2f}')
        # print()


    def getRes_2(self):
        # print('you chose:', 2)
        out='    В исходных данных не предаставлена информация о регионе проживания фрилансеров.'
        print(out)


    def getRes_3(self):
        # print('you chose:', 3)
        prc = self.lowexpertCou / self.expertCou * 100
        tpl='    {prc:.2f}% ( {lcou} из {cou} ) фрилансеров, считающих себя экспертами, выполнил менее 100 проектов'
        out = tpl.format( prc=prc, cou=self.expertCou, lcou=self.lowexpertCou )
        print(out)


    def getRes_4(self):
        # print('you chose:', 1)
        cou = len(self.paymethod)
        print('Всего отмечено {} вид{} оплаты.'
            .format(cou,('' if cou == 1 else 'а' if cou == 0 or cou > 1 and cou < 5 else 'ов')))
        tpl = 'Средний доход у фрилансеров, принимающих оплату в {0:<20}: {1:.2f}'
        for ern in self.paymethod:
            self.erningAvg += self.paymethod[ern]['avg']
            out = tpl.format(ern,self.paymethod[ern]['avg'])
            print(out)


    def getRes_9(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        # print("Load dataset")
        # df = pd.read_csv('/kaggle/input/freelancer-earnings-and-job-trends/freelancer_earnings_bd.csv')
        fn = self.testFilePath
        fn = self.fullFilePath
        df = pd.read_csv(fn)

        print("Basic exploration")
        print(f"Dataset shape:         {df.shape}")
        print(f"Platforms represented: {df['Platform'].nunique()}")
        print(f"Job categories:        {df['Job_Category'].nunique()}")

        # print("Average hourly rate by platform and experience level")
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Platform', y='Hourly_Rate', hue='Experience_Level', data=df)
        plt.title('Average Hourly Rate by Platform and Experience Level')
        plt.xlabel('Platform')
        plt.ylabel('Average Hourly Rate (USD)')
        plt.xticks(rotation=45)
        plt.legend(title='Experience Level')
        plt.tight_layout()
        plt.show()

class Questions:
    title='Freelancer Earnings and Job Trends Dataset'
    desc='''Overview
This comprehensive dataset tracks freelancer earnings and job trends across multiple platforms, experience levels, and job categories. It provides valuable insights into compensation patterns, client preferences, and success metrics in the global freelance marketplace, helping freelancers, businesses, and researchers understand the dynamics of the gig economy.
'''
    qnum = 0
    qs={
        1:'Насколько выше доход у фрилансеров, принимающих оплату в криптовалюте, \n\
    по сравнению с другими способами оплаты?',
        2:'Как распределяется доход фрилансеров в зависимости от региона проживания?',
        3:'Какой процент фрилансеров, считающих себя экспертами, выполнил менее 100 проектов?',
        4:'Какой средний доход у фрилансеров, по каждой валюте?',
        # 9:'Overview',
        0:'( Выход ) (Crtl+C)'
    }
    sel=None
    def __init__(self):
        keys = self.qs.keys()
        self.min = min(keys)
        self.max = max(keys)
    def questions(self):
        for n,q in self.qs.items():
            yield n,q

    def __enter__(self):
        self.qnum = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...
        