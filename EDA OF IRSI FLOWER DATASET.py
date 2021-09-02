#!/usr/bin/env python
# coding: utf-8

# # IRIS FLOWER DATASET EXPLORATORY DATA ANALYSIS
# 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("C:/Users/acer/Downloads/archive/IRIS.csv")


# In[5]:


df.head()


# In[7]:


# check for null values
df.isnull().sum()


# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


df.describe()


# # EDA OF DATASET

# UNIVARIATE ANALYSIS

# Uni means one and variate means variable, so in univariate analysis, there is only one dependable variable. The objective of univariate analysis is to derive the data, define and summarize it, and analyze the pattern present in it. In a dataset, it explores each variable separately. It is possible for two kinds of variables- Categorical and Numerical.
# 
# 

# Some patterns that can be easily identified with univariate analysis are Central Tendency (mean, mode and median), Dispersion (range, variance), Quartiles (interquartile range), and Standard deviation.

# In[12]:


df_setosa=df.loc[df['species']=='Iris-setosa']
df_virginica=df.loc[df['species']=='Iris-virginica']
df_versicolor=df.loc[df['species']=='Iris-versicolor']


# In[13]:


plt.plot(df_setosa['sepal_length'], np.zeros_like(df_setosa['sepal_length']),'o')
plt.plot(df_virginica['sepal_length'], np.zeros_like(df_virginica['sepal_length']),'o')
plt.plot(df_versicolor['sepal_length'], np.zeros_like(df_versicolor['sepal_length']),'o')
plt.xlabel('Sepal length')


# In[14]:


plt.bar(df['species'], df['sepal_length'])
plt.xlabel('Species')


# BIVARIATE ANALYSIS

# Bi means two and variate means variable, so here there are two variables. The analysis is related to cause and the relationship between the two variables.

# In[15]:


sns.FacetGrid(df,hue="species",size=5).map(plt.scatter,"sepal_length","sepal_width").add_legend();


# MULTIVAARIATE ANALYSIS

# Multivariate analysis is required when more than two variables have to be analyzed simultaneously. It is a tremendously hard task for the human brain to visualize a relationship among 4 variables in a graph and thus multivariate analysis is used to study more complex sets of data. Types of Multivariate Analysis include Cluster Analysis, Factor Analysis, Multiple Regression Analysis, Principal Component Analysis, etc. More than 20 different ways to perform multivariate analysis exist and which one to choose depends upon the type of data and the end goal to achieve.

# In[16]:


sns.pairplot(df,hue="species",size=3)


# In[ ]:




