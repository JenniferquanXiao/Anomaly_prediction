import numpy as np
import h5py
import scipy.io
import scipy.stats
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


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

f = open('/mnt/home/f0010173/Sinto_Project/LINE1_201906010630.csv')
#f = open('/Users/quan/Documents/Sinto_Project/SintoProject/LINE1_201906010630.csv')
example=pd.read_csv(f,delimiter=",")
#data1=np.array(example)
col_index=[4,5,6,7,8,10]
data5=np.array(example)[2:,col_index]

f = open('/mnt/home/f0010173/Sinto_Project/LINE1_201907010731.csv')
#f = open('/Users/quan/Documents/Sinto_Project/SintoProject/LINE1_201907010731.csv')
example=pd.read_csv(f,delimiter=",")

#data2=np.array(example)
col_index=[4,5,6,7,8,10]
data6=np.array(example)[2:,col_index]

f = open('/mnt/home/f0010173/Sinto_Project/LINE1_201908010831.csv')
#f = open('/Users/quan/Documents/Sinto_Project/SintoProject/LINE1_201908010831.csv')
example=pd.read_csv(f,delimiter=",")
#data3=np.array(example)
col_index=[4,5,6,7,8,10]
data7=np.array(example)[2:,col_index]

#print(data)
data=np.concatenate((data6,data5,data4,data3,data1,data2),axis=0)

a = np.loadtxt("/mnt/home/f0010173/Sinto_Project/weight.txt")
#a = np.loadtxt("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/result2/weight.txt")

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
#target_1[np.where(target_1<0.7)]=0.7
cyctime=np.array(cyctime)[index_cyc]
target_1=target_1[np.argsort(cyctime)]
integer_encoded_3=integer_encoded_3[np.argsort(cyctime)]
 
 


target_1[np.where(target_1>2)]=2    
target_1[np.where(target_1<0.7)]=0.7   

for j in range(len(target_1)):
    target_1[j]=target_1[j]*a[integer_encoded_3[j]]
                           
#label_1=[]
#for j in range(len(integer_encoded_3)):
#    label_1.append(target_1[j]>perc[dic[integer_encoded_3[j]]])
#label_1=np.array(label_1)


label_1=[]
for j in range(len(integer_encoded_3)):
    label_1.append(target_1[j]>1.3)
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

data_1_t=data7[:,0]
        
integer_encoded_1_t=[]
for i in range(0,data_1_t.shape[0]):
    integer_encoded_1_t.append(char_to_int_1[data_1_t[i]])


#Cycle Number 
cyctime_t=[]
j=0
for i in range(len(integer_encoded_1_t)):
    cyctime_t.append(j)
    if (integer_encoded_1_t[i]==37):
        j=j+1

cyctime_t=np.array(cyctime_t)
cyctime_t=max(cyctime_t)-cyctime_t



running_time_t=data7[:,2]
clean_run_t=[]
for i in range(0, len(running_time_t)):
    clean_run_t.append(float(running_time_t[i]))
#print(clean_run)
    
    
base_time_t=data7[:,3]
clean_base_t=[]
for i in range(0, len(base_time_t)):
    clean_base_t.append(float(base_time_t[i]))
    

data_2_t=data7[:,4]

integer_encoded_2_t=[]
for i in range(0,data_2_t.shape[0]):
    integer_encoded_2_t.append(char_to_int_2[data_2_t[i]])

target_2=[]
for i in range(0,len(clean_run_t)):
    if clean_base_t[i]==0:
        target_2.append(0)
    else:
        target_2.append(clean_run_t[i]/clean_base_t[i])

i=alphabet_1[16]
index_cyc=np.where((np.array(integer_encoded_1_t)== char_to_int_1[i])&(np.array(target_2)!=0))
target_2=np.array(target_2)[index_cyc]
integer_encoded_3_t=np.array(integer_encoded_2_t)[index_cyc]

cyctime_t=np.array(cyctime_t)[index_cyc]
target_2=target_2[np.argsort(cyctime_t)]
integer_encoded_3_t=integer_encoded_3_t[np.argsort(cyctime_t)]



target_2[np.where(target_2>2)]=2
target_2[np.where(target_2<0.7)]=0.7

for j in range(len(target_2)):
    target_2[j]=target_2[j]*a[integer_encoded_3_t[j]]



label_2=[]
for j in range(len(integer_encoded_3_t)):
    label_2.append(target_2[j]>=1.3)
label_2=np.array(label_2)

pos_2=np.where(label_2==1)[0]

pos_3=np.unique(np.concatenate([pos_1,pos_1-1,pos_1-2,pos_1-3,pos_1-4,pos_1-5,
                                pos_1-6,pos_1-7,pos_1-8,pos_1-9,pos_1-10]))
pos_4=np.unique(np.concatenate([pos_2,pos_2-1,pos_2-2,pos_2-3,pos_2-4,pos_2-5,
                                pos_1-6,pos_2-7,pos_2-8,pos_2-9,pos_2-10]))

#######################Train Data & Validation Data################################



prediction_time = 10 
unroll_length = 50
unroll_length_2 = 50
step=5

X1_train=[]
y_train=[]
l=[]
ax_train=[]
bx_train=[]


for index in range(0,len(target_1)-unroll_length-unroll_length_2-prediction_time,step):
    if (len(np.intersect1d(pos_3,range(index,index+unroll_length)))==0):
        ax_train.append(index+unroll_length)
        bx_train.append(range(index,index+unroll_length))
        X1_train.append(target_1[index:index+unroll_length])
        l.append(label_1[index+unroll_length+prediction_time:index+unroll_length+unroll_length_2+prediction_time])



ax_train=np.array(ax_train)
bx_train=np.asarray(bx_train)



#df=np.diff(ax_train)
#df[np.where(df!=1)[0]]=61
#df=np.insert(df,0,ax_train[0])
#ax_train=np.cumsum(df)




X1_train=np.asarray(X1_train)
X2_train=X1_train.copy()


l=np.asarray(l)

y_train=[]
for j in range(l.shape[0]):
    y_train.append(sum(l[j,:])!=0)
y_train=np.asarray(y_train)

print("y_train", y_train.shape)

y2_train=y_train.copy()
######################Test Data##############################



X1_test=[]
y_test=[]
l=[]
ax_test=[]
bx_test=[]



for index in range(0,len(target_2)-unroll_length-unroll_length_2-prediction_time,step):
    if (len(np.intersect1d(pos_4,range(index,index+unroll_length)))==0):
        ax_test.append(index+unroll_length)
        bx_test.append(range(index,index+unroll_length))
        X1_test.append(target_2[index:index+unroll_length])
        l.append(label_2[index+unroll_length+prediction_time:index+unroll_length+unroll_length_2+prediction_time])

ax_test=np.array(ax_test)
bx_test=np.asarray(bx_test)


#df=np.diff(ax_test)
#df[np.where(df!=1)[0]]=61
#df=np.insert(df,0,ax_test[0])
#ax_test=np.cumsum(df)

l=np.asarray(l)
X1_test=np.asarray(X1_test)

y_test=[]
for j in range(l.shape[0]):
    y_test.append(sum(l[j,:])!=0)
y_test=np.asarray(y_test)

print("y_test", y_test.shape)

#mm_scaler = preprocessing.MinMaxScaler()
#X3_train_scaled = mm_scaler.fit_transform(X3_train)
#X3_test_scaled=mm_scaler.transform(X3_test)
#
#X3_train_scaled=X3_train_scaled.reshape(X3_train_scaled.shape[0],X3_train_scaled.shape[1],1)
#X3_test_scaled=X3_test_scaled.reshape(X3_test_scaled.shape[0],X3_test_scaled.shape[1],1)

X1_train, X1_valid, y_train, y_valid = train_test_split(X1_train, y_train, test_size=0.2,random_state=123)

X1_train_y0 = X1_train[y_train==0]
X1_train_y1 = X1_train[y_train==1]
X1_valid_y0 = X1_valid[y_valid==0]
X1_valid_y1 = X1_valid[y_valid==1]

###############################################################

X1_train_y0=X1_train_y0.reshape(X1_train_y0.shape[0],X1_train_y0.shape[1],1)
X1_train=X1_train.reshape(X1_train.shape[0],X1_train.shape[1],1)
X1_valid=X1_valid.reshape(X1_valid.shape[0],X1_valid.shape[1],1)
X1_valid_y0=X1_valid_y0.reshape(X1_valid_y0.shape[0],X1_valid_y0.shape[1],1)
X1_valid_y1=X1_valid_y1.reshape(X1_valid_y1.shape[0],X1_valid_y1.shape[1],1)
X1_test=X1_test.reshape(X1_test.shape[0],X1_test.shape[1],1)
X2_train=X2_train.reshape(X2_train.shape[0],X2_train.shape[1],1)

###################### Model Construction #####################
from keras.layers import Input
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.regularizers import L1L2
from keras.models import model_from_json
import keras
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve
import seaborn as sns
from keras.constraints import UnitNorm, Constraint

visible = Input(shape=(unroll_length,1))
encoder = LSTM(50,activation='relu',return_sequences=False)(visible)
#encoder = LSTM(512, activation='relu',kernel_initializer='random_uniform')(visible)

# define reconstruct decoder
decoder1 = RepeatVector(unroll_length)(encoder)
decoder1 = LSTM(50,activation='relu', return_sequences=True)(decoder1)
#decoder1 = Dropout(0.1)(decoder1)
decoder1 = TimeDistributed(Dense(1))(decoder1)


#json_file = open('/mnt/home/f0010173/Sinto_Project/autoencoder5.json', 'r')
#model_json = json_file.read()
#json_file.close()
#model = model_from_json(model_json)
## load weights into new model
#model.load_weights("/mnt/home/f0010173/Sinto_Project/autoencoder5.h5")
#print("Loaded model from disk")

adams = optimizers.Adam(lr=1e-6)
model = Model(inputs=visible, outputs=decoder1)
#sgd = optimizers.SGD(lr=1e-4, momentum=0.9, nesterov=True)
model.compile(optimizer=adams, loss='mse')


earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=2)

# fit model
history=model.fit(X1_train_y0, X1_train_y0, validation_data=(X1_valid_y0, X1_valid_y0),
                epochs=100,shuffle=True,callbacks=[earlystopper],verbose=2,batch_size=16)
#model.summary()

history_dict = history.history
history_dict.keys()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("autoencoder7_loss.png")
plt.show()


model_json = model.to_json()
with open("autoencoder7.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("autoencoder7.h5")
print("Saved model to disk")



X1_test.tofile("X1_test_7.bin")
X2_train.tofile("X2_train_7.bin")
X1_valid.tofile("X1_valid_7.bin")
np.savetxt("y_valid_7.txt",y_valid)
np.savetxt("y_test_7.txt",y_test)
np.savetxt("y2_train_7.txt",y2_train)


#X1_test_scaled = np.fromfile("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/X1_test_3.bin", dtype=np.float)
#X1_test_scaled.shape=(-1,50,1)
#X2_train_scaled = np.fromfile("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/X2_train_3.bin", dtype=np.float)
#X2_train_scaled.shape=(-1,50,1)
#X1_valid_scaled = np.fromfile("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/X1_valid_3.bin", dtype=np.float)
#X1_valid_scaled.shape=(-1,50,1)
#y_valid = np.loadtxt("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/y_valid_3.txt")
#y_test = np.loadtxt("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/y_test_3.txt")
#y_train = np.loadtxt("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/y2_train_3.txt")
#
#
#json_file = open('/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/autoencoder3.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/autoencoder3.h5")
#print("Loaded model from disk")
#
#
#
####################### validation data ##############################
#
#valid_x_predictions = loaded_model.predict(X1_valid_scaled)
#mse = np.mean(np.power(X1_valid_scaled-valid_x_predictions,2), axis=1)
#
#error_df = pd.DataFrame({'Reconstruction_error': mse.reshape(mse.shape[0],),
#                        'True_class': y_valid.tolist()})
#
#precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
#plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)
#plt.vlines(0.5, 0, 1, colors="r", zorder=100, label='Threshold')
#plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)
#plt.title('Precision and recall for different threshold values')
#plt.xlabel('Threshold')
#plt.ylabel('Precision/Recall')
#plt.legend()
#plt.xlim(0,10)
##plt.savefig("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/result/precision_recall1.png")
#plt.show()
#
################################# test ################################
#test_x_predictions = loaded_model.predict(X1_test_scaled)
#mse = np.mean(np.power(X1_test_scaled-test_x_predictions,2), axis=1)
#
#error_df = pd.DataFrame({'Reconstruction_error': mse.reshape(mse.shape[0],),
#                        'True_class': y_test.tolist()})
#
#threshold_fixed =0.6
#groups = error_df.groupby('True_class')
#fig, ax = plt.subplots()
#
#for name, group in groups:
#    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
#            label= "Break" if name == 1 else "Normal")
#ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
#ax.legend()
#plt.title("Reconstruction error for different classes")
#plt.ylabel("Reconstruction error")
#plt.xlabel("Data point index")
#plt.ylim(0,5)
##plt.savefig("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/result/reconstruction1.png")
#plt.show();
#
########################### Accuracy #############################
#
#LABELS = ["Normal","Break"]
#pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
#conf_matrix = confusion_matrix(error_df.True_class, pred_y)
#
#plt.figure(figsize=(12, 12))
#sns.heatmap(conf_matrix,
#            xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
#plt.title("Confusion matrix")
#plt.ylabel('True class')
#plt.xlabel('Predicted class')
##plt.savefig("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/result/confusionmatrix.png")
#plt.show()
#
#fpr, tpr, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
#roc_auc = auc(fpr, tpr)
#
#plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
#plt.legend(loc='lower right')
#plt.plot([0,1],[0,1],'r--')
#plt.xlim([-0.001, 1])
#plt.ylim([0, 1.001])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
##plt.savefig("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/result/roc.png")
#plt.show();
#
################ test data ###################
#y_pred = np.array(pred_y)
#num=np.where(y_pred==1)[0]
#def group_consecutive(a):
#    return np.split(a, np.where(np.diff(a) != 1)[0] + 1)
#piece_pred=group_consecutive(num)
#
#ind_pred=[]
#for j in range(len(piece_pred)):
#    ind_pred.append([ax_test[piece_pred[j][0]],
#                ax_test[piece_pred[j][-1]]])
#ind_pred=np.asarray(ind_pred)
#
#num=np.where(y_test==1)[0]
#piece_true=group_consecutive(num)
#
#ind_true=[]
#for j in range(len(piece_true)):
#    ind_true.append([ax_test[piece_true[j][0]],
#                ax_test[piece_true[j][-1]]])
#ind_true=np.asarray(ind_true)
#
#################### test data ####################
#fig, axs = plt.subplots(figsize=(13, 5))
#axs.plot(target_2,color='blue', linewidth=1.5,alpha=0.5,label='test data')
#
#
#
#for j in range(ind_pred.shape[0]):
#    axs.hlines(1.27, ind_pred[j,0],ind_pred[j,1],colors='red',linewidth=4,linestyles = "-")
#axs.hlines(1.27, ind_pred[j,0],ind_pred[j,1],colors='red',linewidth=4,linestyles = "-",label='predicted anomaly area')
#
#for j in range(ind_true.shape[0]):
#    axs.hlines(1.3, ind_true[j,0],ind_true[j,1],colors='darkorange',linewidth=4,linestyles = "-")
#axs.hlines(1.3, ind_true[j,0],ind_true[j,1],colors='darkorange',linewidth=4,linestyles = "-",label='true anomaly area')
#
##axs.axvspan(5700, 5800, alpha=0.4, color='red')
#
#plt.legend(loc='best')
#plt.xlabel("Cycle Time")
##plt.xlim(10620,10800)
##plt.ylim(0.9,1.4)
#plt.ylabel("Ratio")
##plt.savefig("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/result/test_3.png")
#plt.show()
#
#
################ train data ###################
#train_x_predictions = loaded_model.predict(X2_train_scaled)
#mse = np.mean(np.power(X2_train_scaled-train_x_predictions, 2), axis=1)
#
#error_df = pd.DataFrame({'Reconstruction_error': mse.reshape(mse.shape[0],),
#                        'True_class': y_train.tolist()})
#
#
#y_pred = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
#y_pred = np.array(y_pred)
#num=np.where(y_pred==1)[0]
#piece_pred=group_consecutive(num)
#
#ind_pred=[]
#for j in range(len(piece_pred)):
#    ind_pred.append([ax_train[piece_pred[j][0]],
#                ax_train[piece_pred[j][-1]]])
#ind_pred=np.asarray(ind_pred)
#
#num=np.where(y2_train==1)[0]
#piece_true=group_consecutive(num)
#
#ind_true=[]
#for j in range(len(piece_true)):
#    ind_true.append([ax_train[piece_true[j][0]],
#                ax_train[piece_true[j][-1]]])
#ind_true=np.asarray(ind_true)
#
######################## train data ##########################
#fig, axs = plt.subplots(figsize=(13, 5))
#axs.plot(target_1,color='blue', linewidth=1.5,alpha=0.3,label='train data')
#
#for j in range(ind_pred.shape[0]):
#    axs.hlines(1.27, ind_pred[j,0],ind_pred[j,1],colors='red',linewidth=4,linestyles = "-")
#axs.hlines(1.27, ind_pred[j,0],ind_pred[j,1],colors='red',linewidth=4,linestyles = "-",label='predicted anomaly area')
#
#for j in range(ind_true.shape[0]):
#    axs.hlines(1.3, ind_true[j,0],ind_true[j,1],colors='darkorange',linewidth=4,linestyles = "-")
#axs.hlines(1.3, ind_true[j,0],ind_true[j,1],colors='darkorange',linewidth=4,linestyles = "-",label='true anomaly area')
#
#
#plt.legend(loc='upper left')
#plt.xlabel("Cycle Time")
#plt.ylabel("Ratio")
##plt.xlim(36000,38000)
##plt.ylim(0.9,1.6)
##plt.savefig("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/result/train_5.png")
#plt.show()

