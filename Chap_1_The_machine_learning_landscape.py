
# coding: utf-8

# In[77]:


import pandas as pd
import sklearn
import os
import numpy as np
from matplotlib import pyplot


# In[64]:


gdp=pd.read_csv("handson-ml/datasets/lifesat/gdp_per_capita.csv", thousands=",",decimal=".", delimiter="\t",encoding="latin-1",na_values="n/a")
oecd_bli=pd.read_csv("handson-ml/datasets/lifesat/oecd_bli_2015.csv",delimiter=",")


# In[67]:


print(oecd_bli["Indicator"].unique())
oecd_bli.head()


# In[112]:


oecd_bli=oecd_bli[oecd_bli['Indicator']=="Life satisfaction"]
data=oecd_bli.merge(gdp,on="Country")
data=data[data["INEQUALITY"]=="TOT"]
data=data.set_index("Country")
data=data[["Value","2015"]]


# In[113]:


data=data.rename(columns={"2015":"GDP","Value":"Life satisfaction"})
data.shape


# In[114]:


model=sklearn.linear_model.LinearRegression(fit_intercept=True,copy_X=True)

model.fit(np.array(data['GDP']).reshape(-1,1),np.array(data['Life satisfaction']).reshape(-1,1))


# In[119]:


print(np.linspace(0,100000,1000).reshape(-1,1).shape)
pyplot.plot(np.linspace(0,100000,1000).reshape(-1,1),model.predict(np.linspace(0,100000,1000).reshape(-1,1)).reshape(-1,1))


# In[120]:


pyplot.scatter(np.array(data['GDP']).reshape(-1,1),np.array(data['Life satisfaction']).reshape(-1,1),color='r')
pyplot.show()


# In[121]:


print(model.coef_,model.intercept_)


# In[122]:


print(model.predict([[22587]]))

