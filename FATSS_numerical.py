# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:50:41 2019

@author: HF
"""


import numpy as np
#from numpy import argmax
import pandas as pd
from time import time
import matplotlib.pyplot as plt

def peptide1hotter(seq):
    aa = "ARNDCQEGHILKMFPSTWYV"
    c2i = dict((c,i) for i,c in enumerate(aa))
    int_encoded = [c2i[char] for char in seq]
    onehot_encoded = list()
    for value in int_encoded:
        letter = [0 for _ in range(len(aa))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return(onehot_encoded)

def one_hot_encoding(x, allowed_set):
   if x not in allowed_set:
       raise Exception(
           'Input {0} not in allowed 1-hot set {1}'.format(x, allowed_set))
   return list(map(lambda s: x == s, allowed_set))


def one_hot_encoding_unk(x, allowed_set):
   if x not in allowed_set:
       x = allowed_set[-1]
   return list(map(lambda s: x == s, allowed_set))

allonehotter = np.vectorize(peptide1hotter)
simmat = pd.read_csv('M4MoS2Training.csv',header=None).values
s2ngs=pd.read_csv('set2_sorted.csv')
s2ngs.set_index('AA_seq',inplace=True)
set2sb = s2ngs[s2ngs['Survive3 ']==1]
sbref1=set2sb.sample(frac=0.8)
testseq=s2ngs[~s2ngs.index.isin(sbref1.index)]
one_hots_ref = []
for i in range(len(sbref1)):
    one_hots_ref.append(peptide1hotter(sbref1.index[i]))
cmat=np.zeros([12,20])
for i in range(len(one_hots_ref)):
    now=time()
    cmat=np.add(cmat,one_hots_ref[i])
    print(i,(time()-now))
TSS=[]
for i in range(len(testseq.index)):
    now=time()
    ssmask=np.dot(np.transpose(peptide1hotter(testseq.index[i])),cmat)
    TSS.append(np.dot(np.ones([1,20]),np.dot(np.multiply(ssmask,simmat),np.ones([20,1]))))
    print(i,time()-now)
#TSS=np.asarray(TSS)
testseq['TSS']=TSS
allstrong=testseq[testseq['Survive3 ']==1]
somestrong=testseq[(testseq['Survive3 ']>=0.10)&(testseq['Survive3 ']<0.3)]
allmedium=testseq[(testseq['Survive3 ']==0)&(testseq['Survive2']<=1)&(testseq['Survive2']>=0.75)]
somemedium=testseq[(testseq['Survive3 ']==0)&(testseq['Survive2']>=0.40)&(testseq['Survive2']<=0.60)]
alllow=testseq[(testseq['Survive2']==0)&(testseq['Survive1']>=0.75)]
vlow=testseq[(testseq['Survive2']==0)&(testseq['Survive1']<=0.40)&(testseq['Survive1']>=0.20)]
vvlow=testseq[testseq['Survive1']==0]
TSS_ss = 0
TSS_sls=0
TSS_sm=0
TSS_slm=0
TSS_sl=0
TSS_svl=0
TSS_svvl=0
for i in range(len(allstrong)):
    TSS_ss = TSS_ss+allstrong.iloc[i,-1]
ASS_ss=TSS_ss/len(allstrong)
for i in range(len(somestrong)):
    TSS_sls = TSS_sls+somestrong.iloc[i,-1]
ASS_sls=TSS_sls/len(somestrong)
for i in range(len(allmedium)):
    TSS_sm = TSS_sm+allmedium.iloc[i,-1]
ASS_sm=TSS_sm/len(allmedium)
for i in range(len(somemedium)):
    TSS_slm = TSS_slm+somemedium.iloc[i,-1]
ASS_slm=TSS_slm/len(somemedium)
for i in range(len(alllow)):
    TSS_sl = TSS_sl+alllow.iloc[i,-1]
ASS_sl=TSS_sl/len(alllow)
for i in range(len(vlow)):
    TSS_svl = TSS_svl+vlow.iloc[i,-1]
ASS_svl=TSS_svl/len(vlow)
for i in range(len(vvlow)):
    TSS_svvl = TSS_svvl+vvlow.iloc[i,-1]
ASS_svvl=TSS_svvl/len(vvlow)
ASS = [ASS_ss,ASS_sls,ASS_sm,ASS_slm,ASS_sl,ASS_svl,ASS_svvl]
xrange = ['ss','sls','sm','slm','sl','svl','svvl']
plt.bar(xrange,height=ASS)
plt.show()