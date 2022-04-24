#!/usr/bin/env python
# coding: utf-8

# In[82]:


import numpy as np
import pandas as pd
import scipy
from math import sqrt
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


# In[83]:


#data
Credit = pd.read_csv('DfRaw.csv')
Credit.head()


# In[84]:


Credit.dtypes


# In[86]:


Credit = pd.get_dummies(Credit, columns=['default payment next month'])


# In[87]:


Credit.columns


# In[88]:


X = Credit.iloc[:,5:11]
X


# In[89]:


y = Credit["default payment next month_not default"]
y


# In[102]:


model = LinearRegression()
msk = np.random.rand(len(Credit)) <0.8
train = Credit[msk]
test = Credit[~msk]


# In[103]:


print (cross_val_score(model,X,y,cv=3))


# In[104]:


algosClass=[]
algosClass.append(("Random Forest Regressor",RandomForestRegressor()))
algosClass.append(("Linear Regression",LinearRegression()))
algosClass.append(("Support Vector Regression",SVR()))


# In[105]:


results=[]
names=[]

for name, model in algosClass:
    result=cross_val_score(model,X,y,cv=3, scoring="r2")
    names.append(name)
    results.append(result)
    
for i in range (len(names)):
    print(names[i],result[i].mean())


# In[106]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)
print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[107]:


algo=RandomForestRegressor()
models=algo.fit(X_train, y_train)
predictions=models.predict(X_test)


# In[108]:


rmse = sqrt(mean_squared_error(y_test, predictions))
predRsquared=r2_score(y_test,predictions)
print ("R squared: %.3f" % predRsquared)
print ("RMSE: %.3f" %rmse)


# In[114]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(X=y_test,  y=predictions)
plt.Xlabel("Ground Truth")
plt.ylabel("Predictions")
plt.show()


# In[115]:


Credit1=pd.DataFrame({"Actual":y_test,"Predicted": predictions})
Credit1


# In[116]:


Credit2 = Credit1.head(25)
Credit2.plot(kind='bar',figsize=(16,7))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[118]:


plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y1_test, color='red', linewidth=2)
plt.show()


# In[ ]:




