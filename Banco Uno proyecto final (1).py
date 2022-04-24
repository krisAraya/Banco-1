#!/usr/bin/env python
# coding: utf-8

# <center> <h1 style="color : #800000"> Banco Uno </h1> </center>
# <center> <h2 style="color : #B22222"> Proyecto final </h2> </center>
# <center> <h3 style="color : #CD5C5C"> Kristel Araya</h3> </center>

# In[14]:


from sqlalchemy import create_engine
from numpy import array
import pymysql
import pandas as pd
import numpy as np    #el manejo de estructuras de datos como listas, directorios, arrays
import matplotlib.pyplot as plt #se utiliza para crear graficas basadas en los datos.
import matplotlib as mplt #se utiliza para crear graficas basadas en datos.


# In[17]:


db_connection_str = 'mysql+pymysql://deepanalytics:Sqltask1234!@34.73.222.197/deepanalytics'

db_connection = create_engine(db_connection_str)
df = pd.read_sql('SELECT * FROM credit', con=db_connection)
df. to_csv ( 'BancoUno.csv')


# In[18]:


df=pd.read_csv ( 'BancoUno.csv')


# In[22]:


df.columns = ["0","ID","LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","default payment next month"]


# In[23]:


df.shape


# In[24]:


DfRaw = df.drop(columns="ID")
DfRaw.head()


# In[25]:


import pandas_profiling
pandas_profiling.ProfileReport(DfRaw)


# In[26]:


DfRaw.columns


# In[27]:


DfRaw = DfRaw.drop_duplicates()


# In[28]:


import pandas_profiling
pandas_profiling.ProfileReport(DfRaw)


# In[39]:


DfRaw = DfRaw.drop(DfRaw.index[[201, 203]])
DfRaw['PAY_0'].value_counts()


# In[40]:


header = DfRaw.dtypes.index
print(header)


# In[41]:


plt.hist(DfRaw['LIMIT_BAL'])
plt.show()


# In[42]:


plt.hist(DfRaw['LIMIT_BAL'], bins=4)


# In[43]:


plt.plot(DfRaw['LIMIT_BAL'])
plt.show()


# In[44]:


x = DfRaw['PAY_0']

y = DfRaw['PAY_2']

plt.scatter(x,y)
plt.show()


# In[45]:


corrMat = DfRaw.corr()
print(corrMat)


# In[46]:


covMat = DfRaw.cov()
print(covMat)


# In[47]:


DfRaw.to_csv("DfRaw.csv")


# In[ ]:




