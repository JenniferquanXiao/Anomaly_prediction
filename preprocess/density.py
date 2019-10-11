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

f = open('/Users/quan/Documents/Sinto_Project/SintoProject/LINE1_201904010430_Copy.csv')
example=pd.read_csv(f,delimiter=",")
data3=np.array(example)
col_index=[4,5,6,7,8,10]
data3=np.array(example)[2:,col_index]

#print(data)
data=np.concatenate((data3,data1,data2),axis=0)

alphabet_1=[]
data_1=data[:,0]
for i in data_1:
    if i not in alphabet_1:
        alphabet_1.append(i)

alphabet_2=[]
data_2=data[:,4]
for i in data_2:
    if i not in alphabet_2:
        alphabet_2.append(i)
        
alphabet_3=[]
data_3=data[:,5]
for i in data_3:
    if i not in alphabet_3:
        alphabet_3.append(i)
        
char_to_int_1 = dict((c, i) for i, c in enumerate(alphabet_1))
int_to_char_1 = dict((i, c) for i, c in enumerate(alphabet_1))
integer_encoded_1=[]
for i in range(0,data_1.shape[0]):
    integer_encoded_1.append(char_to_int_1[data_1[i]])
        
char_to_int_2 = dict((c, i) for i, c in enumerate(alphabet_2))
int_to_char_2 = dict((i, c) for i, c in enumerate(alphabet_2))
integer_encoded_2=[]
for i in range(0,data_2.shape[0]):
    integer_encoded_2.append(char_to_int_2[data_2[i]])
    
    
#import csv

#def dict2csv(dictname={'abc':13,'def':16},filename='filename',key='key',value='value'.encode()):
#    #filename+='.csv'
#    f=open(filename, 'w',encoding='utf8',newline='')
#    writer = csv.writer(f)#创建一个名为sessdefaults的csv文件
#    #writer.writerow([key, value])#指出字段名
#    for key in dictname:
#        item_list = key+":"+str(dictname[key])#将键值对元组转换为列表
#       writer.writerow(item_list)#分行写入
#            #print repr(item_list)
        
#dict2csv(char_to_int_1,"/Users/decslaptop/Documents/Sinto_Project/SintoProject/encode/char_int_1.csv")

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
                          np.array(integer_encoded_2).reshape(len(integer_encoded_2),1),
                          np.array(stickdate).reshape(len(stickdate),1), 
                          np.array(clean_run).reshape(len(clean_run),1), 
                          np.array(clean_base).reshape(len(clean_base),1)), axis=1)

for i in alphabet_1:
    for j in alphabet_2:
        
        index_cyc=np.where((np.array(integer_encoded_1)== char_to_int_1[i]) & (np.array(integer_encoded_2)== char_to_int_2[j]) & (input_all[:,3]!=0) & (input_all[:,4]!=0))
        input_cyc=input_all[index_cyc]
        input_cyc_2 = input_cyc[:,3]
        input_cyc_3 = input_cyc[:,4]
        if(len(input_cyc_3)==0):
            continue
        else:
            target_cyc_1=(input_cyc_2/input_cyc_3)
            #target_cyc_1[np.where(target_cyc_1<2)]=0
            target_cyc_1[np.where(target_cyc_1>2)]=2
            #temp=set(input_cyc_3)
            #if len(temp) == 1:
            #    print(1)
            #else:
            #    print(0)
            plt.hist(target_cyc_1, bins=40, normed=0, facecolor="green", edgecolor="black", alpha=0.7)
            #plt.vlines(1.2, 0, max(z[0])+1,color='red',linestyles = "dashed")
            plt.ylabel('Frequency')
            plt.xlabel('Ratio')
            plt.title('Tag_'+str(char_to_int_1[i])+' Model_'+str(char_to_int_2[j]))
            plt.savefig("/Users/quan/Documents/Sinto_Project/SintoProject/distribution2/distribution_"+str(char_to_int_1[i])+"_"+str(char_to_int_2[j])+".png")
            plt.show()
            
mat=np.ones((44,13))

for i in alphabet_1:
    for j in alphabet_2:
        #member=list()
        index_cyc=np.where((np.array(integer_encoded_1)== char_to_int_1[i]) & (np.array(integer_encoded_2)== char_to_int_2[j]) & (input_all[:,3]!=0) & (input_all[:,4]!=0))
        input_cyc=input_all[index_cyc]
        #input_cyc_1 = input_cyc[:,2]
        input_cyc_2 = input_cyc[:,3]
        input_cyc_3 = input_cyc[:,4]
        if(len(input_cyc_3)==0):
            mat[char_to_int_1[i],char_to_int_2[j]]=0

from openpyxl import Workbook            
def save(data,path):
    wb = Workbook()
    ws = wb.active # 激活 worksheet
    [h, l] = data.shape  # h为行数，l为列数
    for i in range(h):
        row = []
        for j in range(l):
            row.append(data[i,j])
        ws.append(row)
    wb.save(path)

    
save(mat,"/Users/quanp/Documents/Sinto_Project/SintoProject/encode.csv")

    
#for j in alphabet_2:
#    index_cyc=np.where((np.array(integer_encoded_2)== char_to_int_2[j]) & (input_all[:,3]!=0) & (input_all[:,2]!=0))
#    label=input_all[index_cyc]
#    label_1=charlabel[:,1]
    
i=alphabet_1[16]

    
plt.hist(target_1[integer_encoded_3==1], bins=100, normed=0, facecolor="green", edgecolor="black", alpha=0.7)
#plt.vlines(1.2, 0, max(z[0])+1,color='red',linestyles = "dashed")
plt.ylabel('Frequency')
plt.xlabel('Ratio')
plt.title('Tag_'+str(char_to_int_1[i]))
#plt.savefig("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/result2/distribution_"+str(char_to_int_1[i])+".png")
plt.show()

np.mean(target_1)+2.5*np.std(target_1)
