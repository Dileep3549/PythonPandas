
# coding: utf-8

# In[13]:


import os
import pandas as pd
os.getcwd()


# In[14]:


House_details=r"C:\Users\dillu\Documents\Kaggle\Kaggle House\train.csv"


# In[15]:


home_data=pd.read_csv(House_details)


# In[16]:


type(home_data)


# In[21]:


home_data.columns


# In[22]:


home_data.head()


# In[23]:


home_data.describe()


# In[24]:


MeanOfLotArea=round(home_data['LotArea'].mean())


# In[25]:


round(home_data['LotArea'].mean())


# In[26]:


import datetime as dt


# In[27]:


age=dt.datetime.now().year-home_data['YearBuilt'].max()


# In[28]:


age


# In[29]:


## Building a first Model


# In[30]:


from sklearn.tree import DecisionTreeRegressor


# In[35]:


y=home_data.SalePrice


# In[36]:


req_features=['LotArea','1stFlrSF','2ndFlrSF','BedroomAbvGr','YearBuilt','FullBath','TotRmsAbvGrd']


# In[37]:


X=home_data[req_features]


# In[38]:


ModFit=DecisionTreeRegressor(random_state=1993)


# In[39]:


ModFit.fit(X,y)


# In[41]:


Predictions=ModFit.predict(X)


# In[45]:


print(Predictions)

