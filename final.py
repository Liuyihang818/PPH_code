#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[2]:
# read data

df= pd.read_csv(r"C:\Users\Desktop\data.csv",encoding="gbk")
df


# In[3]:


df=df.dropna()
df


# In[4]:


df['出血标注']=0
dropindex=[]
for index,row in df.iterrows():
    if row['产后出血量'] >=450:
        df.loc[index:index,('出血标注')]=1
        #print(df.loc[index:index,('出血标注')])
    else:
        if pd.isna(row['产后出血量']): 
            dropindex.append(index)
            #tempdf.drop(index=index)
df.drop(index=dropindex,inplace=True)
#Delete less useful column data
dropcol=[]

dropcol.append('产后出血量')
dropcol.append('产后出血')
df.drop(axis=1,columns=dropcol,inplace=True)


# In[5]:


from sklearn import preprocessing
tempvalue=df.loc[:,'引产方式'].values.reshape(-1,1)
df.loc[:,'引产方式']= OrdinalEncoder().fit_transform(tempvalue)
tempvalue=df.loc[:,'分娩方式'].values.reshape(-1,1)
df.loc[:,'分娩方式']= OrdinalEncoder().fit_transform(tempvalue)
data_normalized =preprocessing.normalize(df,norm='l1') 
data_normalized


# In[6]:


df


# # random forest

# In[7]:

df = shuffle(df)
df_train,df_test = train_test_split(df,test_size = 0.2)
traindata_y=df_train['出血标注'].values
testdata_y=df_test['出血标注'].values
cols=[]
lens=df.columns.values.__len__()-1
for i in range(lens):
    cols.append(df.columns.values[i])
traindata_x=df_train[cols].values
testdata_x=df_test[cols].values


# In[8]:


# Acquisition of all the training data marked as 1, and 4 times training data marked as 0
def product_model():
    true_flag=df_train[df_train.出血标注==1]
    false_flag=df_train[df_train.出血标注==0]
    tempdata=false_flag.sample(n=3*true_flag.__len__())
    traindata=pd.concat([tempdata,true_flag])
    traindata_Y=traindata['出血标注'].values
    cols=[]
    lens=traindata.columns.values.__len__()-1
    for i in range(lens):
        cols.append(traindata.columns.values[i])
        traindata_x=traindata[cols].values
    rand_model = RandomForestClassifier()#随机森林模型
    rand_model.fit(traindata_x,traindata_Y)
    return rand_model


# In[9]:


# Generating multiple prediction models
randnames=[]
for i in range(1,102,1):
    randname=product_model()
    randnames.append(randname)


# In[10]:


# Use the generated multiple prediction models to predict the test data by soft voting（10%）.
# As long as more than 10% of the models recognize a record of massive bleeding, it is predicted to be massive bleeding.
votepercent=0.01
predictedvaules=[]
testdata_preY=testdata_y
for randname in randnames:
    predicted_y=randname.predict(testdata_x)
    predictedvaules.append(predicted_y)
length=testdata_x.__len__()
model_length=predictedvaules.__len__()
testdata_preY=[]
for i in range(0,length):
    bloodnum=0
    nonbloodnum=0
    for j in range(0, model_length):
        if predictedvaules[j][i]==1:
            bloodnum=bloodnum+1
        else:
            nonbloodnum=nonbloodnum+1
    if bloodnum/(bloodnum+nonbloodnum) >=votepercent:
        testdata_preY.append(1)
    else:
        testdata_preY.append(0)


#the prediction results of the test data are analyzed to look at the accuracy rate, accuracy rate, and recall rate
testdata_preY=np.array(testdata_preY)
accuracy_value=accuracy_score(testdata_y, testdata_preY)
print(accuracy_value)
precison_value=precision_score(testdata_y, testdata_preY)
print(precison_value)
recall_value=recall_score(testdata_y, testdata_preY)
print(recall_value)
f_value=(precison_value*recall_value*2)/(precison_value+recall_value)
print(f_value)


# In[11]:



print(testdata_preY.sum())
print(testdata_y.sum())


# #KNN

# In[ ]:


df= pd.read_csv(r"C:\Users\yihan\Desktop\final.csv",encoding="gbk")
df
df=df.dropna()
df
df['出血标注']=0
dropindex=[]
for index,row in df.iterrows():
    if row['产后出血量'] >=450:
        df.loc[index:index,('出血标注')]=1
        #print(df.loc[index:index,('出血标注')])
    else:
        if pd.isna(row['产后出血量']): 
            dropindex.append(index)
            #tempdf.drop(index=index)
df.drop(index=dropindex,inplace=True)
#Delete less useful column data
dropcol=[]

dropcol.append('产后出血量')
dropcol.append('产后出血')
df.drop(axis=1,columns=dropcol,inplace=True)
from sklearn import preprocessing
tempvalue=df.loc[:,'引产方式'].values.reshape(-1,1)
df.loc[:,'引产方式']= OrdinalEncoder().fit_transform(tempvalue)
tempvalue=df.loc[:,'分娩方式'].values.reshape(-1,1)
df.loc[:,'分娩方式']= OrdinalEncoder().fit_transform(tempvalue)
dn=df
del dn["出血标注"]
data_normalized =preprocessing.normalize(dn,norm='l1') 
data_normalized


# In[ ]:


from sklearn import neighbors
y=df['出血标注']
traindata_x,testdata_x,traindata_y,testdata_y = train_test_split(data_normalized,y,test_size=0.2,random_state=42)


# In[ ]:


# Acquisition of all the training data marked as 1, and 4 times training data marked as 0
def product_model():
    true_flag=df_train[df_train.出血标注==1]
    false_flag=df_train[df_train.出血标注==0]
    tempdata=false_flag.sample(n=3*true_flag.__len__())
    traindata=pd.concat([tempdata,true_flag])
     
    traindata_Y=traindata['出血标注'].values
     
    cols=[]
    lens=traindata.columns.values.__len__()-1
    for i in range(lens):
        cols.append(traindata.columns.values[i])
        traindata_x=traindata[cols].values
    classifier = neighbors.KNeighborsClassifier() 
    classifier.fit(traindata_x,traindata_Y)
    return classifier


# In[ ]:


# Generating multiple prediction models
randnames=[]
for i in range(1,100,1):
    randname=product_model()
    randnames.append(randname)


# In[ ]:


# Use the generated multiple prediction models to predict the test data by soft voting（10%）.
# As long as more than 10% of the models recognize a record of massive bleeding, it is predicted to be massive bleeding.
votepercent=0.01
predictedvaules=[]
testdata_preY=testdata_y
for randname in randnames:
    predicted_y=randname.predict(testdata_x)
    predictedvaules.append(predicted_y)
length=testdata_x.__len__()
model_length=predictedvaules.__len__()
testdata_preY=[]
for i in range(0,length):
    bloodnum=0
    nonbloodnum=0
    for j in range(0, model_length):
        if predictedvaules[j][i]==1:
            bloodnum=bloodnum+1
        else:
            nonbloodnum=nonbloodnum+1
    if bloodnum/(bloodnum+nonbloodnum) >=votepercent:
        testdata_preY.append(1)
    else:
        testdata_preY.append(0)


 
 
accuracy_value=accuracy_score(testdata_y, testdata_preY)
print(accuracy_value)
precison_value=precision_score(testdata_y, testdata_preY)
print(precison_value)
recall_value=recall_score(testdata_y, testdata_preY)
print(recall_value)
f_value=(precison_value*recall_value*2)/(precison_value+recall_value)
print(f_value)


# In[ ]:


#svm
from sklearn.svm import SVC


# In[ ]:


# Acquisition of all the training data marked as 1, and 4 times training data marked as 0
def product_model():
    true_flag=df_train[df_train.出血标注==1]
    false_flag=df_train[df_train.出血标注==0]
    tempdata=false_flag.sample(n=3*true_flag.__len__())
    traindata=pd.concat([tempdata,true_flag])
    traindata_Y=traindata['出血标注'].values
    cols=[]
    lens=traindata.columns.values.__len__()-1
    for i in range(lens):
        cols.append(traindata.columns.values[i])
        traindata_x=traindata[cols].values
    classifier =SVC(kernel='rbf', class_weight='balanced')
    classifier.fit(traindata_x,traindata_Y)
    return classifier


# In[ ]:


# Generating multiple prediction models
randnames=[]
for i in range(1,100,1):
    randname=product_model()
    randnames.append(randname)


# In[ ]:


# Use the generated multiple prediction models to predict the test data by soft voting（10%）.
# As long as more than 10% of the models recognize a record of massive bleeding, it is predicted to be massive bleeding.
votepercent=0.01
predictedvaules=[]
testdata_preY=testdata_y
for randname in randnames:
    predicted_y=randname.predict(testdata_x)
    predictedvaules.append(predicted_y)
length=testdata_x.__len__()
model_length=predictedvaules.__len__()
testdata_preY=[]
for i in range(0,length):
    bloodnum=0
    nonbloodnum=0
    for j in range(0, model_length):
        if predictedvaules[j][i]==1:
            bloodnum=bloodnum+1
        else:
            nonbloodnum=nonbloodnum+1
    if bloodnum/(bloodnum+nonbloodnum) >=votepercent:
        testdata_preY.append(1)
    else:
        testdata_preY.append(0)

 
 
 
accuracy_value=accuracy_score(testdata_y, testdata_preY)
print(accuracy_value)
precison_value=precision_score(testdata_y, testdata_preY)
print(precison_value)
recall_value=recall_score(testdata_y, testdata_preY)
print(recall_value)
f_value=(precison_value*recall_value*2)/(precison_value+recall_value)
print(f_value)


# #KNN

# In[ ]:


#PCA
from sklearn.decomposition import PCA
def getPCAData(data,comp):
    pcaClf = PCA(n_components=comp, whiten=True)
    pcaClf.fit(data)
    data_PCA = pcaClf.transform(data)  
    return data_PCA

def modiData(data):
    x1 = []
    x2=[]
    for i in range(0,len(data+1)):
        x1.append(data[i][0])
        x2.append(data[i][1])
    x1=np.array(x1)
    x2=np.array(x2)
    X=np.array(list(zip(x1,x2))).reshape(len(x1),2)
    return X


# In[ ]:


import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
from mpl_toolkits.mplot3d import Axes3D 


# In[ ]:


# Euclidean distance  
def distEclud(x,y):
    return np.sqrt(np.sum((x-y)**2))  
 
 
def randCent(dataSet,k):
    m,n = dataSet.shape
    centroids = np.zeros((k,n))
    for i in range(k):
        index = int(np.random.uniform(0,m)) #
        centroids[i,:] = dataSet[index,:]
    return centroids
 
def MYKMeans(dataSet,k):
 
    m = np.shape(dataSet)[0] 
    clusterAssment = np.mat(np.zeros((m,2)))
    clusterChange = True
 
    # initialise centroids
    centroids = randCent(dataSet,k)
    while clusterChange:
        clusterChange = False
  
        for i in range(m):
            minDist = 100000.0
            minIndex = -1
 
    
          
            for j in range(k):           
                distance = distEclud(centroids[j,:],dataSet[i,:])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
          
            if clusterAssment[i,0] != minIndex:
                clusterChange = True
                clusterAssment[i,:] = minIndex,minDist**2
    
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]]   
            centroids[j,:] = np.mean(pointsInCluster,axis=0)  
 
    return centroids,clusterAssment


# In[ ]:


def show3D(PCAData,k,centroids,clusterAssment):
    m,n = PCAData.shape
    mark = ['blue','red', 'green', 'yellow']
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    for i in range(m):
        markIndex = int(clusterAssment[i,0])
        ax.scatter(PCAData[:,0:1][i],PCAData[:,1:2][i],PCAData[:,2:3][i],c=mark[markIndex])
    for i in range(k):
        ax.scatter(centroids[i,0],centroids[i,1],marker="x",c="black") 
    plt.show() 

PCAData=getPCAData(data_normalized,3)
centroids,clusterAssment = MYKMeans(PCAData,2)
show3D(PCAData,2,centroids,clusterAssment)


# In[ ]:




