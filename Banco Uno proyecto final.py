#!/usr/bin/env python
# coding: utf-8

# <center> <h1 style="color : #800000"> Banco Uno </h1> </center>
# <center> <h2 style="color : #B22222"> Proyecto final </h2> </center>
# <center> <h3 style="color : #CD5C5C"> Kristel Araya</h3> </center>

# In[139]:


from sqlalchemy import create_engine
from numpy import array
import pymysql
import pandas as pd
import numpy as np    #el manejo de estructuras de datos como listas, directorios, arrays
import matplotlib.pyplot as plt #se utiliza para crear graficas basadas en los datos.
import matplotlib as mplt #se utiliza para crear graficas basadas en datos.


# In[140]:


df. to_csv ( 'BancoUno.csv')


# In[141]:


df.columns = ["ID","LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","default payment next month"]


# In[142]:


df.shape


# In[143]:


DfRaw = df.drop(columns="ID")
DfRaw.head()


# In[144]:


import pandas_profiling
pandas_profiling.ProfileReport(DfRaw)


# In[145]:


DfRaw.columns


# In[147]:


DfRaw = DfRaw.drop_duplicates()


# In[55]:


import pandas_profiling
pandas_profiling.ProfileReport(DfRaw)


# In[148]:


header = DfRaw.dtypes.index
print(header)


# In[149]:


plt.hist(DfRaw['LIMIT_BAL'])
plt.show()


# In[71]:


plt.hist(DfRaw['LIMIT_BAL'], bins=4)


# In[73]:


plt.plot(DfRaw['LIMIT_BAL'])
plt.show()


# In[75]:


x = DfRaw['PAY_0']

y = DfRaw['PAY_2']

plt.scatter(x,y)
plt.show()


# In[150]:


corrMat = DfRaw.corr()
print(corrMat)


# In[151]:


covMat = DfRaw.cov()
print(covMat)


# In[152]:


DfRaw.to_csv("DfRaw.csv")

