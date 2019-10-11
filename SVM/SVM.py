import numpy as np
import h5py
import scipy.io
import scipy.stats
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn import preprocessing
from keras.utils import np_utils

from entropy import spectral_entropy

#git clone https://github.com/raphaelvallat/entropy.git entropy/
#cd entropy/
#pip install -r requirements.txt
#python setup.py develop


#import sys
#sys.stdout.flush()

np.random.seed(1337) # for reproducibility


# Read in sample data



f = open('/mnt/home/f0010173/Sinto_Project/SampleData-Copy.csv')
#f = open('/Users/quan/Documents/Sinto_Project/SintoProject/SampleData-Copy.csv')
example=pd.read_csv(f,delimiter=",")
#data1=np.array(example)
col_index=[4,5,6,7,8,10]
data1=np.array(example)[2:,col_index]

f = open('/mnt/home/f0010173/Sinto_Project/LINE1_2019020120190228-copy.csv')
#f = open('/Users/quan/Documents/Sinto_Project/SintoProject/LINE1_2019020120190228-copy.csv')
example=pd.read_csv(f,delimiter=",")

#data2=np.array(example)
col_index=[4,5,6,7,8,10]
data2=np.array(example)[2:,col_index]

f = open('/mnt/home/f0010173/Sinto_Project/LINE1_201904010430_Copy.csv')
#f = open('/Users/quan/Documents/Sinto_Project/SintoProject/LINE1_201904010430_Copy.csv')
example=pd.read_csv(f,delimiter=",")
#data3=np.array(example)
col_index=[4,5,6,7,8,10]
data3=np.array(example)[2:,col_index]

f = open('/mnt/home/f0010173/Sinto_Project/LINE1_201905010531-copy.csv')
#f = open('/Users/quan/Documents/Sinto_Project/SintoProject/LINE1_201905010531-copy.csv')
example=pd.read_csv(f,delimiter=",")
#data4=np.array(example)
col_index=[4,5,6,7,8,10]
data4=np.array(example)[2:,col_index]

#print(data)
data=np.concatenate((data3,data1,data2),axis=0)

# Tag
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
for i in integer_encoded_1:
    cyctime.append(j)
    if ((i+1)%35==0):
        j=j+1
cyctime=cyctime[::-1]



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
#target_1[np.where(target_1<0.7)]=0.7
cyctime=np.array(cyctime)[index_cyc]
target_1=target_1[np.argsort(cyctime)]
integer_encoded_3=integer_encoded_3[np.argsort(cyctime)]
integer_encoded_3[np.where((integer_encoded_3==10)|(integer_encoded_3==11)
|(integer_encoded_3==12))]=10  
    
onehot_encoded_3=np_utils.to_categorical(integer_encoded_3,
                                         max(np.unique(integer_encoded_3)+1))
onehot_encoded_3=np.delete(onehot_encoded_3,6,axis=1)

target_1[np.where(target_1>1.5)]=1.5     
target_1[np.where(target_1<0.7)]=0.7  
    
perc=[]
aver=[]
for j in np.unique(integer_encoded_3):
    idx=np.where(integer_encoded_3== j)[0]
    perc.append(np.percentile(target_1[idx], 98))
    aver.append(np.mean(target_1))

dic = dict((c, i) for i, c in enumerate(np.unique(integer_encoded_3)))           
                           
#label_1=[]
#for j in range(len(integer_encoded_3)):
#    label_1.append(target_1[j]>perc[dic[integer_encoded_3[j]]])
#label_1=np.array(label_1)



for j in range(len(target_1)):
    target_1[j]=target_1[j]-aver[dic[integer_encoded_3[j]]]

target_3=target_1+aver[0]


label_1=[]
for j in range(len(integer_encoded_3)):
    label_1.append(target_3[j]>1.27)
label_1=np.array(label_1)

pos_1=np.where(label_1==1)[0]


######################Test Data##############################

#fig, axs = plt.subplots(figsize=(20, 5)) 
#axs.plot(y_train,color='blue', label='y_train',linewidth=1.5)
#plt.legend(loc='upper left')
#plt.xlabel("Cycle Time")
#plt.ylabel("Ratio")
#plt.ylim(0.7,1.5)
##plt.savefig("/Users/quan/Desktop/modelchange.png")
#plt.show()

betabet_1=[]
data_1_t=data4[:,0]
for i in data_1_t:
    if i not in betabet_1:
        betabet_1.append(i)
        
char_to_int_1_t = dict((c, i) for i, c in enumerate(betabet_1))
int_to_char_1_t = dict((i, c) for i, c in enumerate(betabet_1))
integer_encoded_1_t=[]
for i in range(0,data_1_t.shape[0]):
    integer_encoded_1_t.append(char_to_int_1_t[data_1_t[i]])


#Cycle Number 
cyctime_t=[]
j=0
for i in integer_encoded_1_t:
    cyctime_t.append(j)
    if ((i+1)%35==0):
        j=j+1
cyctime_t=cyctime_t[::-1]



running_time_t=data4[:,2]
clean_run_t=[]
for i in range(0, len(running_time_t)):
    clean_run_t.append(float(running_time_t[i]))
#print(clean_run)
    
    
base_time_t=data4[:,3]
clean_base_t=[]
for i in range(0, len(base_time_t)):
    clean_base_t.append(float(base_time_t[i]))
    
betabet_2=[]
data_2_t=data4[:,4]
for i in data_2_t:
    if i not in betabet_2:
        betabet_2.append(i)
        
char_to_int_2_t = dict((c, i) for i, c in enumerate(betabet_2))
int_to_char_2_t = dict((i, c) for i, c in enumerate(betabet_2))
integer_encoded_2_t=[]
for i in range(0,data_2_t.shape[0]):
    integer_encoded_2_t.append(char_to_int_2_t[data_2_t[i]])

target_2=[]
for i in range(0,len(clean_run_t)):
    if clean_base_t[i]==0:
        target_2.append(0)
    else:
        target_2.append(clean_run_t[i]/clean_base_t[i])

i=alphabet_1[16]
index_cyc=np.where((np.array(integer_encoded_1_t)== char_to_int_1_t[i])&(np.array(target_2)!=0))
target_2=np.array(target_2)[index_cyc]
integer_encoded_3_t=np.array(integer_encoded_2_t)[index_cyc]
target_2[np.where(target_2>1.5)]=1.5
target_2[np.where(target_2<0.7)]=0.7
cyctime_t=np.array(cyctime_t)[index_cyc]
target_2=target_2[np.argsort(cyctime_t)]
integer_encoded_3_t=integer_encoded_3_t[np.argsort(cyctime_t)]

integer_encoded_3_t[np.where((integer_encoded_3_t==10)|(integer_encoded_3_t==11)
|(integer_encoded_3_t==12))]=10   
    
integer_encoded_3_t[np.where(integer_encoded_3_t==6)]= 0   

onehot_encoded_3_t=np_utils.to_categorical(integer_encoded_3_t,
                                         max(np.unique(integer_encoded_3_t)+1))
onehot_encoded_3_t=np.delete(onehot_encoded_3_t,6,axis=1)
onehot_encoded_3_t=np.concatenate((onehot_encoded_3_t,
                                  np.zeros(onehot_encoded_3_t.shape[0]).reshape(
                                          onehot_encoded_3_t.shape[0],1)),axis=1)

#perc_t=[]
#aver_t=[]
#for j in np.unique(integer_encoded_3_t):
#    idx=np.where(integer_encoded_3_t== j)[0]
#    perc_t.append(np.percentile(target_2[idx], 98))
#    aver_t.append(np.mean(target_2<(np.percentile(target_2[idx], 98))))
    

#dic = dict((c, i) for i, c in enumerate(np.unique(integer_encoded_3_t)))           

#!!!!Model 6 doesn't exist in train set
                           
#label_2=[]
#for j in range(len(integer_encoded_3_t)):
#    if integer_encoded_3_t[j]==6:
#        label_2.append(target_2[j]>1.2)
#    else:
#        label_2.append(target_2[j]>perc[dic[integer_encoded_3_t[j]]])
#label_2=np.array(label_2)
    


for j in range(len(target_2)):
    target_2[j]=target_2[j]-aver[dic[integer_encoded_3_t[j]]]
    
target_4=target_2+aver[0]

label_2=[]
for j in range(len(integer_encoded_3_t)):
    label_2.append(target_4[j]>=1.27)
label_2=np.array(label_2)

pos_2=np.where(label_2==1)[0]


#######################Train Data & Validation Data################################



prediction_time = 10 
unroll_length = 50
unroll_length_2 = 50

X1_train=[]
X2_train=[]
y_train=[]
l=[]
raw_train=[]
ax_train=[]
ay_train=[]

for index in range(0,len(target_1)-unroll_length-unroll_length_2-prediction_time,prediction_time):
    if len(np.intersect1d(pos_1,range(index,index+unroll_length)))==0:
        ax_train.append(range(index+unroll_length+prediction_time,index+unroll_length+2*prediction_time))
        raw_train.append(target_3[index+unroll_length+prediction_time:index+unroll_length+2*prediction_time])
        X1_train.append(target_1[index:index+unroll_length])
        X2_train.append(onehot_encoded_3[index,:])
        ay_train.append(index+unroll_length+prediction_time)
        l.append(label_1[index+unroll_length+prediction_time:index+unroll_length+unroll_length_2+prediction_time])
if len(np.intersect1d(pos_1,range(index,index+unroll_length)))==0:
    ax_train.append(range(index+unroll_length+2*prediction_time,index+unroll_length+unroll_length_2+prediction_time))
    raw_train.append(target_3[index+unroll_length+2*prediction_time:index+unroll_length+unroll_length_2+prediction_time])


raw_train=np.concatenate(raw_train)
ax_train=np.concatenate(ax_train)
ay_train=np.asarray(ay_train)
by_train=np.where(np.in1d(ax_train,ay_train))[0]


#df=np.diff(ax_train)
#df[np.where(df!=1)[0]]=61
#df=np.insert(df,0,ax_train[0])
#ax_train=np.cumsum(df)




X1_train=np.asarray(X1_train)
X2_train=np.asarray(X2_train)

l=np.asarray(l)

y_train=[]
for j in range(l.shape[0]):
    y_train.append(sum(l[j,:])!=0)
y_train=np.asarray(y_train)



f_1=np.mean(X1_train,axis=1)
f_2=np.median(X1_train,axis=1).reshape(X1_train.shape[0],1)
f_3=np.std(X1_train,axis=1).reshape(X1_train.shape[0],1)
f_4=np.var(X1_train,axis=1).reshape(X1_train.shape[0],1)
f_5=np.sqrt(np.mean(X1_train**2,axis=1)).reshape(X1_train.shape[0],1)
f_6=np.mean(np.gradient(X1_train,axis=1),axis=1).reshape(X1_train.shape[0],1)
f_7=scipy.stats.skew(X1_train,axis=1).reshape(X1_train.shape[0],1)
f_8=scipy.stats.kurtosis(X1_train,axis=1).reshape(X1_train.shape[0],1)
f_9=scipy.stats.iqr(X1_train,axis=1).reshape(X1_train.shape[0],1)
f_10=(np.sum(np.diff(np.transpose(np.transpose(X1_train)-f_1)>0,axis=1)!=0,axis=1)/unroll_length).reshape(X1_train.shape[0],1)
f_1=f_1.reshape(X1_train.shape[0],1)
def fun(a): 
    # Returning the sum of elements at start index and at last index 
    # inout array 
     return spectral_entropy(a,100,normalize=True,method='welch') 

f_11=np.apply_along_axis(fun,1,X1_train).reshape(X1_train.shape[0],1)
X3_train=np.concatenate((f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10,f_11,X2_train),axis=1)

print("X3_train", X3_train.shape)
#print("X2_train", X2_train.shape)
print("y_train", y_train.shape)


######################Test Data##############################



X1_test=[]
X2_test=[]
y_test=[]
l=[]
raw_test=[]
ax_test=[]
ay_test=[]


for index in range(0,len(target_2)-unroll_length-unroll_length_2-prediction_time,prediction_time):
    if len(np.intersect1d(pos_2,range(index,index+unroll_length)))==0:
        ax_test.append(range(index+unroll_length+prediction_time,index+unroll_length+2*prediction_time))
        raw_test.append(target_4[index+unroll_length+prediction_time:index+unroll_length+2*prediction_time])
        X1_test.append(target_2[index:index+unroll_length])
        X2_test.append(onehot_encoded_3_t[index,:])
        ay_test.append(index+unroll_length+prediction_time)
        l.append(label_2[index+unroll_length+prediction_time:index+unroll_length+unroll_length_2+prediction_time])
if len(np.intersect1d(pos_2,range(index,index+unroll_length)))==0:
    ax_test.append(range(index+unroll_length+2*prediction_time,index+unroll_length+unroll_length_2+prediction_time))
    raw_test.append(target_4[index+unroll_length+2*prediction_time:index+unroll_length+unroll_length_2+prediction_time])
    
raw_test=np.concatenate(raw_test)
ax_test=np.concatenate(ax_test)
ay_test=np.asarray(ay_test)
by_test=np.where(np.in1d(ax_test,ay_test))[0]

#df=np.diff(ax_test)
#df[np.where(df!=1)[0]]=61
#df=np.insert(df,0,ax_test[0])
#ax_test=np.cumsum(df)

l=np.asarray(l)
X1_test=np.asarray(X1_test)
X2_test=np.asarray(X2_test)

y_test=[]
for j in range(l.shape[0]):
    y_test.append(sum(l[j,:])!=0)
y_test=np.asarray(y_test)


f_1=np.mean(X1_test,axis=1)
f_2=np.median(X1_test,axis=1).reshape(X1_test.shape[0],1)
f_3=np.std(X1_test,axis=1).reshape(X1_test.shape[0],1)
f_4=np.var(X1_test,axis=1).reshape(X1_test.shape[0],1)
f_5=np.sqrt(np.mean(X1_test**2,axis=1)).reshape(X1_test.shape[0],1)
f_6=np.mean(np.gradient(X1_test,axis=1),axis=1).reshape(X1_test.shape[0],1)
f_7=scipy.stats.skew(X1_test,axis=1).reshape(X1_test.shape[0],1)
f_8=scipy.stats.kurtosis(X1_test,axis=1).reshape(X1_test.shape[0],1)
f_9=scipy.stats.iqr(X1_test,axis=1).reshape(X1_test.shape[0],1)
f_10=(np.sum(np.diff(np.transpose(np.transpose(X1_test)-f_1)>0,axis=1)!=0,axis=1)/unroll_length).reshape(X1_test.shape[0],1)
f_1=f_1.reshape(X1_test.shape[0],1)

f_11=np.apply_along_axis(fun,1,X1_test).reshape(X1_test.shape[0],1)
X3_test=np.concatenate((f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10,f_11,X2_test),axis=1)


print("X3_test", X3_test.shape)
#print("X2_train", X2_train.shape)
print("y_test", y_test.shape)

mm_scaler = preprocessing.MinMaxScaler()
X3_train_scaled = mm_scaler.fit_transform(X3_train)
X3_test_scaled=mm_scaler.transform(X3_test)


#################### Cross Validation split #############

ind1=np.where(y_train==1)[0]
ind2=np.where(y_train==0)[0]
d1=int(len(ind1)/5)
d2=int(len(ind2)/5)
v1=X3_train_scaled[np.sort(np.concatenate((ind1[0:d1],ind2[0:d2])))]
v2=X3_train_scaled[np.sort(np.concatenate((ind1[d1:2*d1],ind2[d2:2*d2])))]
v3=X3_train_scaled[np.sort(np.concatenate((ind1[2*d1:3*d1],ind2[2*d2:3*d2])))]
v4=X3_train_scaled[np.sort(np.concatenate((ind1[3*d1:4*d1],ind2[3*d2:4*d2])))]
v5=X3_train_scaled[np.sort(np.concatenate((ind1[4*d1:],ind2[4*d2:])))]
X4_train_scaled=np.concatenate((v1,v2,v3,v4,v5),axis=0)
v1=y_train[np.sort(np.concatenate((ind1[0:d1],ind2[0:d2])))]
v2=y_train[np.sort(np.concatenate((ind1[d1:2*d1],ind2[d2:2*d2])))]
v3=y_train[np.sort(np.concatenate((ind1[2*d1:3*d1],ind2[2*d2:3*d2])))]
v4=y_train[np.sort(np.concatenate((ind1[3*d1:4*d1],ind2[3*d2:4*d2])))]
v5=y_train[np.sort(np.concatenate((ind1[4*d1:],ind2[4*d2:])))]
y2_train=np.concatenate((v1,v2,v3,v4,v5),axis=0)


###################### Model Construction #####################
import pandas as pd
from sklearn.svm import SVC

import scikitplot as skplt
#pip install scikit-plot==0.3.1
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs



svc = SVC(kernel='rbf',probability=True,class_weight={0:1,1:6},random_state=0,
          gamma=0.13)



sfs1 = SFS(svc, 
           k_features=11, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='f1',
           n_jobs=-1,
           cv=5)

sfs = sfs1.fit(X3_train_scaled, y_train)

fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
plt.ylim([0, 0.3])
plt.title('Sequential Forward Selection (w. StdDev)')
plt.ylabel("Cross validation f1-score")
plt.grid()
plt.savefig("/Users/quan/Documents/Sinto_Project/report/selection.png")
plt.show()

print('Selected features:', sfs1.k_feature_idx_)
print(sfs.k_score_)

X_train_sfs = sfs1.transform(X3_train_scaled)
X_test_sfs = sfs1.transform(X3_test_scaled)

# Fit the estimator using the new feature subset
# and make a prediction on the test data
svc.fit(X_train_sfs, y_train)
y_pred = svc.predict(X_test_sfs)
y_probs=svc.predict_proba(X_test_sfs)

# Compute the accuracy of the prediction
recall = float(sum(y_test[np.where(y_test==1)] == y_pred[np.where(y_test==1)])) / len(np.where(y_test==1)[0])
print('Test set recall: %.2f %%' % (recall * 100))

precision = float(sum(y_test[np.where(y_pred==1)] == y_pred[np.where(y_pred==1)])) / len(np.where(y_pred==1)[0])
print('Test set precision: %.2f %%' % (precision * 100))


#skplt.metrics.plot_confusion_matrix(y_test, y_pred)

skplt.metrics.plot_precision_recall_curve(y_test, y_probs)
plt.savefig("prcurve2.png")
plt.show()

print(confusion_matrix(y_test, y_pred))
f1_score(y_test, y_pred, average='micro')

############### test data ###################
num=np.where(y_pred==1)[0]
def group_consecutive(a):
    return np.split(a, np.where(np.diff(a) != 1)[0] + 1)
piece_pred=group_consecutive(num)

ind_pred=[]
for j in range(len(piece_pred)):
    ind_pred.append([by_test[piece_pred[j][0]],
                by_test[piece_pred[j][-1]]])
ind_pred=np.asarray(ind_pred)

num=np.where(y_test==1)[0]
piece_true=group_consecutive(num)

ind_true=[]
for j in range(len(piece_true)):
    ind_true.append([by_test[piece_true[j][0]],
                by_test[piece_true[j][-1]]])
ind_true=np.asarray(ind_true)

################### test data ####################
fig, axs = plt.subplots(figsize=(13, 5))
axs.plot(raw_test,color='blue', linewidth=1.5,alpha=0.5,label='test data')

for j in range(ind_pred.shape[0]):
    axs.hlines(1.27, ind_pred[j,0],ind_pred[j,1],colors='red',linewidth=4,linestyles = "-")
axs.hlines(1.27, ind_pred[j,0],ind_pred[j,1],colors='red',linewidth=4,linestyles = "-",label='predicted anomaly area')

for j in range(ind_true.shape[0]):
    axs.hlines(1.3, ind_true[j,0],ind_true[j,1],colors='orange',linewidth=4,linestyles = "-")
axs.hlines(1.3, ind_true[j,0],ind_true[j,1],colors='orange',linewidth=4,linestyles = "-",label='true anomaly area')


plt.legend(loc='best')
plt.xlabel("Cycle Time")
#plt.xlim(4210,4500)
plt.ylabel("Ratio")
plt.savefig("/Users/quan/Documents/Sinto_Project/report/test.png")
plt.show()

################### train data ######################
y_pred = svc.predict(X_train_sfs)
y_probs=svc.predict_proba(X_train_sfs)

# Compute the accuracy of the prediction
recall = float(sum(y_train[np.where(y_train==1)] == y_pred[np.where(y_train==1)])) / len(np.where(y_train==1)[0])
print('Train set recall: %.2f %%' % (recall * 100))

precision = float(sum(y_train[np.where(y_pred==1)] == y_pred[np.where(y_pred==1)])) / len(np.where(y_pred==1)[0])
print('Train set precision: %.2f %%' % (precision * 100))


#skplt.metrics.plot_confusion_matrix(y_train, y_pred)

skplt.metrics.plot_precision_recall_curve(y_train, y_probs)
plt.savefig("prcurve_train.png")
plt.show()

print(confusion_matrix(y_train, y_pred))

############### train data ###################
num=np.where(y_pred==1)[0]
def group_consecutive(a):
    return np.split(a, np.where(np.diff(a) != 1)[0] + 1)
piece_pred=group_consecutive(num)

ind_pred=[]
for j in range(len(piece_pred)):
    ind_pred.append([by_train[piece_pred[j][0]],
                by_train[piece_pred[j][-1]]])
ind_pred=np.asarray(ind_pred)

num=np.where(y_train==1)[0]
piece_true=group_consecutive(num)

ind_true=[]
for j in range(len(piece_true)):
    ind_true.append([by_train[piece_true[j][0]],
                by_train[piece_true[j][-1]]])
ind_true=np.asarray(ind_true)

################### train data ####################
fig, axs = plt.subplots(figsize=(13, 5))
axs.plot(raw_train,color='blue', linewidth=1.5,alpha=0.3,label='train data')

for j in range(ind_pred.shape[0]):
    axs.hlines(1.27, ind_pred[j,0],ind_pred[j,1],colors='red',linewidth=4,linestyles = "-")
axs.hlines(1.27, ind_pred[j,0],ind_pred[j,1],colors='red',linewidth=4,linestyles = "-",label='predicted anomaly area')

for j in range(ind_true.shape[0]):
    axs.hlines(1.3, ind_true[j,0],ind_true[j,1],colors='orange',linewidth=4,linestyles = "-")
axs.hlines(1.3, ind_true[j,0],ind_true[j,1],colors='orange',linewidth=4,linestyles = "-",label='true anomaly area')


plt.legend(loc='best')
plt.xlabel("Cycle Time")
#plt.xlim(37500,38000)
plt.ylabel("Ratio")
plt.savefig("/Users/quan/Documents/Sinto_Project/report/train.png")
plt.show()

