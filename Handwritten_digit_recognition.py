#!/usr/bin/env python
# coding: utf-8

# # Fetching MNIST dataset

# In[1]:


from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784') 


# # To ignore unnecessary warnings

# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Analysing Data

# In[3]:


x , y = mnist['data'] , mnist['target']


# In[4]:


x


# In[5]:


y


# # Taking a demo data

# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt


# In[7]:


demo_x=x[2874:2875]
demo_ximage = demo_x.values.reshape(28,28)


# In[8]:


plt.imshow(demo_ximage,cmap=matplotlib.cm.binary,interpolation="nearest")
plt.axis('off')


# In[9]:


y[2874]


# # Splitting x data

# In[10]:


x_train,x_test=x[:10000],x[10000:11000]


# # Splitting y data

# In[11]:


y_train,y_test=y[:10000],y[10000:11000]


# # Randomly shuffling the datapoints(it prevents the model from learning the order of the training)

# In[12]:


import numpy as np
from sklearn.utils import shuffle
x_train,y_train = shuffle(x_train,y_train)


# # We will try each of the following classifiers and check their accuracy
# 
# ### 1. DecisionTreeClassifier
# ### 2. LogisticRegression
# ### 3. LinearDiscriminantAnalysis
# ### 4. KNeighborsClassifier
# ### 5. GaussianNB
# ### 6. SVC
# ### 7. MLPClassifier
# ### 8. RandomForestClassifier
# ### 9. AdaBoostClassifier
# ### 10. QuadraticDiscriminantAnalysis

# # DecisionTree Classifier

# In[13]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)
from sklearn.model_selection import cross_val_score
acc = cross_val_score(clf, x_train, y_train, cv=3, scoring="accuracy")
acc.mean()


# # LogisticRegression Classifier

# In[14]:


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(x_train,y_train)
from sklearn.model_selection import cross_val_score
acc = cross_val_score(clf, x_train, y_train, cv=3, scoring="accuracy")
acc.mean()


# # LinearDiscriminantAnalysis Classifier

# In[15]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf=LinearDiscriminantAnalysis()
clf.fit(x_train,y_train)
from sklearn.model_selection import cross_val_score
acc = cross_val_score(clf, x_train, y_train, cv=3, scoring="accuracy")
acc.mean()


# # KNeighbors Classifier

# In[16]:


from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier()
clf.fit(x_train,y_train)
from sklearn.model_selection import cross_val_score
acc = cross_val_score(clf, x_train, y_train, cv=3, scoring="accuracy")
acc.mean()


# # GaussianNB Classifier

# In[17]:


from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(x_train,y_train)
from sklearn.model_selection import cross_val_score
acc = cross_val_score(clf, x_train, y_train, cv=3, scoring="accuracy")
acc.mean()


# # SupportVectorMachine Classifier

# In[18]:


from sklearn.svm import SVC
clf=SVC()
clf.fit(x_train,y_train)
from sklearn.model_selection import cross_val_score
acc = cross_val_score(clf, x_train, y_train, cv=3, scoring="accuracy")
acc.mean()


# # MLP Classifier

# In[19]:


from sklearn.neural_network import MLPClassifier
clf=MLPClassifier()
clf.fit(x_train,y_train)
from sklearn.model_selection import cross_val_score
acc = cross_val_score(clf, x_train, y_train, cv=3, scoring="accuracy")
acc.mean()


# # RandomForest Classifier

# In[20]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(x_train,y_train)
from sklearn.model_selection import cross_val_score
acc = cross_val_score(clf, x_train, y_train, cv=3, scoring="accuracy")
acc.mean()


# # AdaBoost Classifier

# In[21]:


from sklearn.ensemble import AdaBoostClassifier
clf=AdaBoostClassifier()
clf.fit(x_train,y_train)
from sklearn.model_selection import cross_val_score
acc = cross_val_score(clf, x_train, y_train, cv=3, scoring="accuracy")
acc.mean()


# # QuadraticDiscriminantAnalysis Classifier

# In[22]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf=QuadraticDiscriminantAnalysis()
clf.fit(x_train,y_train)
from sklearn.model_selection import cross_val_score
acc = cross_val_score(clf, x_train, y_train, cv=3, scoring="accuracy")
acc.mean()


# # To check the prediction for the demo element

# In[23]:


# clf.predict(demo_x) 


# ### Conclusion: We have used several classifiers and observed accuracy for each one  
