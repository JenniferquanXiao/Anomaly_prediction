import matplotlib.pyplot as plt
import ruptures as rpt
import numpy as np
import h5py
import scipy.io
import scipy.stats
import pandas as pd
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from entropy import spectral_entropy


np.random.seed(1337) # for reproducibility




# Read in sample data



#f = open('/mnt/home/f0010173/Sinto_Project/SampleData-Copy.csv')
f = open('/Users/quan/Documents/Sinto_Project/SintoProject/SampleData-Copy.csv')
example=pd.read_csv(f,delimiter=",")
#data1=np.array(example)
col_index=[4,5,6,7,8,10]
data1=np.array(example)[2:,col_index]

#f = open('/mnt/home/f0010173/Sinto_Project/LINE1_2019020120190228-copy.csv')
f = open('/Users/quan/Documents/Sinto_Project/SintoProject/LINE1_2019020120190228-copy.csv')
example=pd.read_csv(f,delimiter=",")

#data2=np.array(example)
col_index=[4,5,6,7,8,10]
data2=np.array(example)[2:,col_index]

#f = open('/mnt/home/f0010173/Sinto_Project/LINE1_201904010430_Copy.csv')
f = open('/Users/quan/Documents/Sinto_Project/SintoProject/LINE1_201904010430_Copy.csv')
example=pd.read_csv(f,delimiter=",")
#data3=np.array(example)
col_index=[4,5,6,7,8,10]
data3=np.array(example)[2:,col_index]

#f = open('/mnt/home/f0010173/Sinto_Project/LINE1_201905010531-copy.csv')
f = open('/Users/quan/Documents/Sinto_Project/SintoProject/LINE1_201905010531-copy.csv')
example=pd.read_csv(f,delimiter=",")
#data4=np.array(example)
col_index=[4,5,6,7,8,10]
data4=np.array(example)[2:,col_index]

#f = open('/mnt/home/f0010173/Sinto_Project/LINE1_201906010630.csv')
f = open('/Users/quan/Documents/Sinto_Project/SintoProject/LINE1_201906010630.csv')
example=pd.read_csv(f,delimiter=",")
#data1=np.array(example)
col_index=[4,5,6,7,8,10]
data5=np.array(example)[2:,col_index]

#f = open('/mnt/home/f0010173/Sinto_Project/LINE1_201907010731.csv')
f = open('/Users/quan/Documents/Sinto_Project/SintoProject/LINE1_201907010731.csv')
example=pd.read_csv(f,delimiter=",")

#data2=np.array(example)
col_index=[4,5,6,7,8,10]
data6=np.array(example)[2:,col_index]

#f = open('/mnt/home/f0010173/Sinto_Project/LINE1_201908010831.csv')
f = open('/Users/quan/Documents/Sinto_Project/SintoProject/LINE1_201908010831.csv')
example=pd.read_csv(f,delimiter=",")
#data3=np.array(example)
col_index=[4,5,6,7,8,10]
data7=np.array(example)[2:,col_index]

#data4=data7.copy()


#print(data)
data=np.concatenate((data7,data6,data5,data4,data3,data1,data2),axis=0)

#data4=data7.copy()
alphabet_1=[]
data_1=data[:,0]
for i in data_1:
    if i not in alphabet_1:
        alphabet_1.append(i)
        
char_to_int_1 = dict((c, i) for i, c in enumerate(alphabet_1))
int_to_char_1 = dict((i, c) for i, c in enumerate(alphabet_1))
integer_encoded_1=[]
for i in range(0,data_1.shape[0]):
    integer_encoded_1.append(char_to_int_1[data_1[i]])


#Cycle Number 
cyctime=[]
j=0
for i in range(len(integer_encoded_1)):
    cyctime.append(j)
    if (integer_encoded_1[i]==37):
        j=j+1
        
cyctime=np.array(cyctime)
cyctime=max(cyctime)-cyctime




running_time=data[:,2]
clean_run=[]
for i in range(0, len(running_time)):
    clean_run.append(float(running_time[i]))
#print(clean_run)
    
    
base_time=data[:,3]
clean_base=[]
for i in range(0, len(base_time)):
    clean_base.append(float(base_time[i]))

target_1=[]
for i in range(0,len(clean_run)):
    if clean_base[i]==0:
        target_1.append(0)
    else:
        target_1.append(clean_run[i]/clean_base[i])



alphabet_2=[]
data_2=data[:,4]
for i in data_2:
    if i not in alphabet_2:
        alphabet_2.append(i)
        
char_to_int_2 = dict((c, i) for i, c in enumerate(alphabet_2))
int_to_char_2 = dict((i, c) for i, c in enumerate(alphabet_2))
integer_encoded_2=[]
for i in range(0,data_2.shape[0]):
    integer_encoded_2.append(char_to_int_2[data_2[i]])
    



i=alphabet_1[16]
index_cyc=np.where((np.array(integer_encoded_1)== char_to_int_1[i])&(np.array(target_1)!=0))
target_1=np.array(target_1)[index_cyc]
integer_encoded_3=np.array(integer_encoded_2)[index_cyc]
base=np.array(clean_base)[index_cyc]
#target_1[np.where(target_1<0.7)]=0.7
cyctime=np.array(cyctime)[index_cyc]

target_1=target_1[np.argsort(cyctime)]
base=base[np.argsort(cyctime)]
integer_encoded_3=integer_encoded_3[np.argsort(cyctime)]

target_1[np.where(target_1>2)]=2     
target_1[np.where(target_1<0.7)]=0.7  
                               

#label_1=[]
#for j in range(len(integer_encoded_3)):
#    label_1.append(target_1[j]>perc[dic[integer_encoded_3[j]]])
#label_1=np.array(label_1)



modelchange=[]
j=1
for i in range(len(integer_encoded_3)-1):
    modelchange.append(j)
    if (integer_encoded_3[i]!=integer_encoded_3[i+1]):
        j=1
    else:
        j=0
modelchange.append(1)
modelchange=np.array(modelchange)
modelchange=np.where(modelchange==1)[0]
modelchange[-1]+=1

ix=np.where(integer_encoded_3[modelchange[0:len(modelchange)-1]]==1)[0]
ix=np.insert(ix,0,104)
ix=np.append(ix,3739)
mc=modelchange[ix]

integer_encoded_4=integer_encoded_3.copy()
integer_encoded_4[np.where((integer_encoded_4==0)|(integer_encoded_4==2)|(integer_encoded_4==3)
|(integer_encoded_4==4)|(integer_encoded_4==5)|(integer_encoded_4==6)
|(integer_encoded_4==9)|(integer_encoded_4==10)|(integer_encoded_4==11)
|(integer_encoded_4==7))[0]]=1

import sympy as sp
#i=0
for i in range(13):
    locals()['a'+str(i)]=sp.symbols("a"+str(i))
    locals()['b'+str(i)]=sp.symbols("b"+str(i))

a0=1
b0=0

f=0
for j in range(1,len(mc)-1,1):
    m1=np.median(target_1[mc[j]-min(10,mc[j]-mc[j-1]):mc[j]])
    m2=np.median(target_1[mc[j]:mc[j]+min(10,mc[j+1]-mc[j])])
    if max(m1,m2)<1.5:
        f=f+pow(m1*locals()['a'+str(integer_encoded_3[mc[j]-1])]-m2*locals()['a'+str(integer_encoded_3[mc[j]])],2)



f=sympy.simplify(f)
for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
    locals()['b'+str(i)]=sp.diff(f,locals()['a'+str(i)])

r=sp.solve([b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12])

a=[]
for j in range(13):
    if (j==1):
        a.append(1)
    else:
        if ((j==7)|(j==8)|(j==12)):
            a.append(1)
        else:
            a.append(r[locals()['a'+str(j)]])
a[7]=a[6]

target_3=target_1.copy()
for j in range(len(target_1)):
    target_3[j]=target_1[j]*a[integer_encoded_3[j]]

#########################################

modelchange2=[]
j=1
for i in range(len(integer_encoded_4)-1):
    modelchange2.append(j)
    if (integer_encoded_4[i]!=integer_encoded_4[i+1]):
        j=1
    else:
        j=0
modelchange2.append(1)
modelchange2=np.array(modelchange2)
modelchange2=np.where(modelchange2==1)[0]
modelchange2[-1]+=1

ix=np.where(integer_encoded_4[modelchange2[0:len(modelchange2)-1]]==1)[0]
ix=np.append(ix,45)
mc2=modelchange2[ix]

for i in [1,8,12]:
    locals()['a'+str(i)]=sp.symbols("a"+str(i))
    locals()['b'+str(i)]=sp.symbols("b"+str(i))

a1=1
b1=0

f=0
for j in range(1,len(mc2)-1,1):
    m1=np.median(target_3[mc2[j]-min(10,mc2[j]-mc2[j-1]):mc2[j]])
    m2=np.median(target_3[mc2[j]:mc2[j]+min(10,mc2[j+1]-mc2[j])])
    if max(m1,m2)<1.5:
        f=f+pow(m1*locals()['a'+str(integer_encoded_4[mc2[j]-1])]-m2*locals()['a'+str(integer_encoded_4[mc2[j]])],2)



f=sympy.simplify(f)
for i in [8,12]:
    locals()['b'+str(i)]=sympy.diff(f,locals()['a'+str(i)])

r2=sp.solve([b8,b12])

a[8]=r2[a8]
a[12]=r2[a12]

for j in range(len(target_1)):
    target_3[j]=target_1[j]*a[integer_encoded_3[j]]
##########################################
c=a.copy()
c[1]=a[0]
c[0]=a[1]
c[3]=a[2]
c[5]=a[3]
c[6]=a[5]
c[2]=a[6]
c[9]=a[7]
c[7]=a[9]
a=c.copy()
np.savetxt("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/result2/weight.txt",a)




fig, axs = plt.subplots(figsize=(13, 5))
axs.plot(target_1)
plt.xlabel("Cycle Time")
plt.ylabel("Ratio")
#plt.xlim(0,20000)
#plt.savefig("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/result/train2_1.png")
plt.show()