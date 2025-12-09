#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('diabetes.csv')
df


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


x=df.iloc[:,df.columns!='Outcome']
y=df.iloc[:,df.columns=='Outcome']


# In[6]:


y


# In[7]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)


# In[8]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtrain,ytrain.values.ravel())


# In[9]:


output=model.predict(xtest)
output


# In[10]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(output,ytest)
print("The accuracy score is: ",acc)


# In[ ]:




