#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("garments_worker_productivity.csv")


# In[3]:


#statistical summary of the data
df.describe()


# In[4]:


#information about the data
df.info()


# In[5]:


df.head()


# In[6]:


df.drop(['idle_men', 'idle_time','no_of_style_change'], axis=1,inplace=True)
df.head()


# In[7]:


df["department"]= [ "sewing" if i=="sweing" else "finishing" for i in df.department]
df["department"].value_counts()


# In[8]:


df1 = df.copy()


# In[9]:


df.department.value_counts().plot.pie(autopct='%.2f %%')
plt.title("Percentage of each Department")


# In[10]:


sns.countplot(x="actual_productivity",data=df1)
plt.title("Count of Productivity range")


# In[11]:


df["department"].value_counts()


# In[12]:


x= df.corr()
x


# In[13]:


sns.heatmap(x)


# In[14]:


sns.distplot(df["actual_productivity"])
plt.title("Density Plot of Actual Productivity")


# In[15]:


sns.boxplot(x="department",y="no_of_workers",data=df)
plt.title("Average Number of Workers in each Department")


# In[16]:


f,ax = plt.subplots(2, 2, figsize=(8,6))

sns.barplot(data=df1, x='department', y='smv', ax=ax[0][0])
sns.barplot(data=df1, x='quarter', y='incentive', ax=ax[0][1], estimator=np.sum)
sns.barplot(data=df1, x='team', y='over_time', ax=ax[1][0])
sns.barplot(data=df1, x='department', y='wip', ax=ax[1][1])

plt.tight_layout()


# In[17]:


f, ax = plt.subplots(2, 2, figsize=(8,6))

sns.barplot(data=df1, x='quarter', y='no_of_workers',ax=ax[0][0])
sns.barplot(data=df1, x='quarter', y='over_time', ax=ax[0][1])
sns.barplot(data=df1, x='smv', y='over_time', ax=ax[1][0])
sns.barplot(data=df1, x='department', y='over_time', ax=ax[1][1])

plt.tight_layout()


# In[18]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
sns.boxplot(x="actual_productivity",y="day",data=df,ax=ax1)
sns.barplot(data=df, x='day', y='wip',ax=ax2)


plt.tight_layout()


# In[19]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
sns.barplot(data=df, x='team', y='smv', hue='department',ax=ax1)
sns.barplot(data=df, x='day',y='incentive', hue='department',ax=ax2)
plt.tight_layout()


# # LOGISTIC REGRESSION

# In[20]:


df1 = df.copy()


# In[21]:


df1["actual_productivity"]= [ 1 if i>0.60 else 0 for i in df1.actual_productivity]
df1["actual_productivity"].value_counts()


# In[22]:


x=df1[['team', 'targeted_productivity',
       'smv', 'over_time', 'incentive', 'no_of_workers']]
y=df1['actual_productivity']


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30, random_state=101)


# In[25]:


from sklearn.linear_model import LogisticRegression


# In[26]:


log=LogisticRegression()


# In[27]:


log.fit(x_train,y_train)


# In[28]:


predict=log.predict(x_test)


# In[29]:


predict


# In[30]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predict)


# In[31]:


from sklearn.metrics import classification_report


# In[32]:


print(classification_report(y_test,predict))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




