#!/usr/bin/env python
# coding: utf-8

# importing libraries

# In[10]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt  ###maplotlib visualzation
import seaborn as sns ####seaborn visualization


# In[11]:


plt


# Preprocessing

# 

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


application_train=pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\sreekanth\application_train.csv")
# homeCredit_columns_description=pd.read_csv("C:/Users/Lenovo/Downloads/HomeCredit_columns_description.csv",encoding='latin1')


# In[ ]:


application_train.head(5)


# In[6]:


application_train.describe()


# In[7]:


application_train.columns


# In[8]:


application_train.shape


# In[11]:


application_train["TARGET"].value_counts() 


# In[12]:


application_train['TARGET'].value_counts(normalize=True)*100


# In[13]:


application_train.shape


# In[14]:


application_train.describe()


# In[15]:


application_train.isna().sum()


# In[16]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt  ###maplotlib visualzation
import seaborn as sns ####seaborn visualization


# In[7]:


application_train.shape


# In[19]:


application_train.head(8)


# In[20]:


application_train.tail(5)


# In[21]:


application_train.isna().sum()


# In[22]:


application_train["TARGET"].value_counts()


# In[23]:


application_train["TARGET"].value_counts(normalize=True)*100


# In[24]:


application_train.describe()


# In[25]:


application_train.describe()


# In[26]:


application_train


# In[27]:


application_train.isna().sum()


# In[28]:


application_train.isnull().sum()


# In[29]:


application_train


# In[30]:


application_train_cleaned_columns = application_train.dropna(axis=1)
application_train_cleaned_columns


# In[31]:


application_train.isna().sum()


# In[32]:


application_train.fillna(0).abs


# In[38]:


numerical_columns = application_train.select_dtypes(include=np.number).columns
application_train[numerical_columns] = application_train[numerical_columns].fillna(application_train[numerical_columns].mean())


# In[39]:


categorical_columns = application_train.select_dtypes(include='object').columns
application_train[categorical_columns] = application_train[categorical_columns].fillna(application_train[categorical_columns].mode().iloc[0])


# In[40]:


final_nan_count = application_train.isna().sum()


# In[41]:


print(final_nan_count)


# In[42]:


numerical_columns=application_train.select_dtypes(include=np.number).columns
application_train[numerical_columns]=application_train[numerical_columns].fillna(application_train[numerical_columns]).mean()


# In[43]:


categorical_columns=application_train.select_dtypes(include="object").columns
application_train[categorical_columns]=application_train[categorical_columns].fillna(application_train[categorical_columns]).mode().iloc[0]


# In[44]:


final_nan_count=application_train.isna().sum()


# In[45]:


print(final_nan_count)


# In[46]:


application_train.isna().sum()


# In[47]:


application_train


# In[48]:


final_nan_count


# In[49]:


categorical_columns


# In[50]:


application_train[categorical_columns]


# In[51]:


application_train[categorical_columns]=application_train[categorical_columns].fillna(application_train[categorical_columns]).mode().iloc[0]


# In[52]:


print(application_train[categorical_columns])


# In[53]:


application_train.isna().sum()


# In[54]:


application_train[["NAME_FAMILY_STATUS"]].describe()


# In[55]:


application_train[["TARGET"]].describe()


# In[56]:


df=sns.load_dataset("anscombe")


# In[58]:


df.head(5)


# In[60]:


df.shape


# In[61]:


df.isna().sum()


# In[62]:


df.describe()


# In[63]:


import matplotlib as plt


# In[66]:


sns.barplot()


# In[70]:


sns.boxenplot()


# In[69]:


sns.axes_style()


# In[72]:


sns.categorical


# In[ ]:




