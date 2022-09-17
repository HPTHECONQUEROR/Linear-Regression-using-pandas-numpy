#!/usr/bin/env python
# coding: utf-8

# # LINEAR REGRESSION BETWEEN TV &  SALES

# IMPORTING DEPENDENCIES

# In[102]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# READING THE CSV FILE AND SAVING IT IN VARIABLE d

# In[126]:


d=pd.read_csv('mark.csv')


# FETCHING THE COLUMN DATA OF TV,Sales TO x,y AS NUMPY ARRAYS

# In[127]:


x=np.array(d['TV'])
y=np.array(d['Sales'])


# NOW WE ARE SPLITTING THE DATA OF X,Y OF 80%/20% FOR BOTH TRAING AND TESTING

# In[105]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2)


# RESHAPING THE VALUES OF TEST AND TRAIN DATA TO (-1,1) 
# HERE -1 DENOTES EVERY SINGLE ELEMENT NSIDE THE VARIABLE
# AND 1 SAYS THE DATA TO HAVE A SINGLE COLUMN ONLY

# In[106]:


X_train=X_train.reshape(-1,1)
Y_train=Y_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)
Y_test=Y_test.reshape(-1,1)


# CREATING A MODEL FOR LINEAR REGRESSION AND STORING IT IN MOD VARIABLE

# In[107]:


mod=LinearRegression()


# NOW FITTING ALL THE TRAING DATA USING FIT FUNCTION

# In[108]:


mod.fit(X_train,Y_train)


# # PREDICTING THE VALUES OF Y_TEST

# In[123]:


mod.predict(X_test)
Y_test


# IN THE ABOVE WE CAN UNDERSTAND THAT THE EROOR IS MORE BCOZ WE HAVE ONLY LESS NUMBER OF DATA IF THE NUMBER OF DATA INCREASES THE MACHINE USED TO LEARN MORE AND IT WONT HAVE MORE DEVIATIONS

# # PLOTTING THE BEST FIT

# In[125]:


plt.scatter(X_train,Y_train,color='LIGHTGREEN')
plt.plot(X_train,mod.predict(X_train),color='GREY')


# In[ ]:





# In[ ]:





# In[ ]:




