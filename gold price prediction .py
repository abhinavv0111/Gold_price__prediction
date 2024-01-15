#!/usr/bin/env python
# coding: utf-8

# # project on gold price prediction

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[2]:


#loading the csv data into a pandas data frame


# In[3]:


gold_data = pd.read_csv('gld_price_data.csv')


# In[4]:


gold_data.head(10)


# In[5]:


gold_data.tail()


# In[6]:


gold_data.shape


# In[7]:


gold_data.info()


# In[8]:


gold_data.isnull().sum()


# In[9]:


gold_data.describe()


# In[10]:


correlation = gold_data.corr()


# In[11]:


plt.figure(figsize=(8,8))


# In[12]:


plt.figure(figsize = (8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True,annot_kws={'size':8},cmap='Blues') 


# In[13]:


gold_data.corr()


# In[14]:


print(correlation['GLD'])


# In[15]:


sns.distplot(gold_data['GLD'],color = 'green')


# In[16]:


X = gold_data.drop(['Date','GLD'], axis=1)


# In[17]:


Y = gold_data['GLD']


# In[18]:


print(X)


# In[19]:


print(Y)


# In[20]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 2)


# In[21]:


regressor = RandomForestRegressor(n_estimators = 100)


# In[22]:


regressor.fit(X_train , Y_train)


# In[23]:


test_data_prediction = regressor.predict(X_test)


# In[24]:


print(test_data_prediction)


# In[25]:


error_score = metrics.r2_score(Y_test , test_data_prediction)


# In[26]:


print ("R squared error:", error_score)


# In[27]:


Y_test = list(Y_test)


# In[ ]:





# In[28]:


plt.plot(Y_test , color = 'blue', label = 'Actual Value')
plt.plot(test_data_prediction , color = 'green', label = 'Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')

plt.show()


# In[ ]:




