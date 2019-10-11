#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import h5py
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
np.random.seed(1337) # for reproducibility


# # Read in sample data

# In[2]:

f = open('/Users/quan/Documents/Sinto_Project/SintoProject/SampleData-Copy.csv')
example=pd.read_csv(f,delimiter=",")
data1=np.array(example)
col_index=[4,5,6,7,8,10]
data1=np.array(example)[2:,col_index]

f = open('/Users/quan/Documents/Sinto_Project/SintoProject/LINE1_2019020120190228-copy.csv')
example=pd.read_csv(f,delimiter=",")
data2=np.array(example)
col_index=[4,5,6,7,8,10]
data2=np.array(example)[2:,col_index]

f = open('/Users/quan/Documents/Sinto_Project/SintoProject/LINE1_2019020120190228-copy.csv')
example=pd.read_csv(f,delimiter=",")
data3=np.array(example)
col_index=[4,5,6,7,8,10]
data3=np.array(example)[2:,col_index]

#print(data)
data=np.concatenate((data1,data2,data3),axis=0)

#生成类别搜索引擎
alphabet_1=[]
data_1=data[:,0]
for i in data_1:
    if i not in alphabet_1:
        alphabet_1.append(i)
#print(alphabet_1)
        
#对类别搜索引擎进行标号，并生成dict以便索引
char_to_int_1 = dict((c, i) for i, c in enumerate(alphabet_1))
int_to_char_1 = dict((i, c) for i, c in enumerate(alphabet_1))
integer_encoded_1=[]
for i in range(0,data_1.shape[0]):
    integer_encoded_1.append(char_to_int_1[data_1[i]])
#print(integer_encoded_1)

#one-hot encoding    
onehot_encoded_1 = list()
for value in integer_encoded_1:
    letter = [0 for _ in range(len(alphabet_1))]
    letter[value] = 1
    onehot_encoded_1.append(letter)
#print(onehot_encoded_1)

#process the time
import time
def transfer_date(date):
    if date[1]!='/':
        month=10*int(date[0])+int(date[1])
        i=3
    else:
        month=int(date[0])
        i=2
    if date[i+1]!='/':
        day=10*int(date[i])+int(date[i+1])
        i=i+3
    else:
        day=int(date[i])
        i=i+2
    year=int(date[i:i+4])
    i=i+5
    if date[i+1]!=':':
        hour=int(date[i])*10+int(date[i+1])
        i=i+3
    else:
        hour=int(date[i])
        i=i+2
    minute=int(date[i:i+2])

    return(year, month, day, hour, minute, 0, 0, 0, 0)
def timestick(date):
    a=transfer_date(date)
    return(time.mktime(a))


# In[6]:


date=data[:,1]
#print(date)
stickdate=[]
for i in range(0, len(date)):
    stickdate.append(timestick(date[i]))
#print(stickdate)

running_time=data[:,2]
clean_run=[]
for i in range(0, len(running_time)):
    clean_run.append(float(running_time[i]))
#print(clean_run)
    
    
base_time=data[:,3]
clean_base=[]
for i in range(0, len(base_time)):
    clean_base.append(float(base_time[i]))


input_all=np.concatenate((np.array(integer_encoded_1).reshape(len(integer_encoded_1),1),
                          np.array(stickdate).reshape(len(integer_encoded_1),1), 
                          np.array(clean_run).reshape(len(integer_encoded_1),1), 
                          np.array(clean_base).reshape(len(integer_encoded_1),1)), axis=1)



for j in alphabet_1[0:37]:
    #member=list()
    index_cyc=np.where((np.array(integer_encoded_1)== char_to_int_1[j]) & (input_all[:,3]!=0) & (input_all[:,2]!=0))
    input_cyc=input_all[index_cyc]
    input_cyc_1 = np.sort(input_cyc[:,1])
    ind = np.argsort(input_cyc[:,1])
    input_cyc_2 = input_cyc[ind,2]
    input_cyc_3 = input_cyc[ind,3]
    
    #print(input_cyc_1)
    target_cyc_1=(input_cyc_2/input_cyc_3)
    #target_cyc_1[np.where(target_cyc_1<2)]=0
    input_cyc_2[np.where(target_cyc_1>2)]=2*input_cyc_3[np.where(target_cyc_1>2)]
    input_cyc_2=input_cyc_2[np.where (np.array(target_cyc_1)>0)]
    #input_cyc_3=input_cyc_3[np.where((np.array(target_cyc_1)<=2))]
    locals()['ts'+str(char_to_int_1[j])]=input_cyc_2




#plt.bar(range(len(ts4)),ts4,color='green')
#plt.show()
#plt.legend() # 显示图例
#plt.ylabel('rate')
#plt.xlabel('time')
#plt.ylim(0 , 1.2)
#plt.savefig("/Users/quan/Documents/Sinto_Project/SintoProject/LINE1_2019020120190228.png")


#import matplotlib.cm as cm
#colors=cm.rainbow(np.linspace(0,1,len(alphabet_1)))

#plt.figure(figsize=(8, 6)) 

for i in range(37):
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(locals()['ts'+str(i)])),locals()['ts'+str(i)],color='green')
    plt.savefig("/Users/quan/Documents/Sinto_Project/SintoProject/ts"+str(i)+".png")
    plt.show()
#plt.legend() # 显示图例
#plt.ylabel('rate')
#plt.xlabel('time')
#plt.ylim(0 , 1.2)
#plt.savefig("/Users/quan/Documents/Sinto_Project/SintoProject/LINE1_2019020120190228.png")
#plt.show()
