#!/usr/bin/env python
# coding: utf-8

# # GRIP: The Sparks Foundation

# # Data Science and Business Analytics Internship

# # Author: Suchandra Majumder

# # Task 1: Prediction Using Supervised Machine Learning

# In this task it is required to predict the percentage of a student on the basis of number of hours studied using the Linear Regression supervised machine learning algorithm.

# # Importing Modules:

# In[21]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# # Reading and Observing Data

# In[22]:


url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df= pd.read_csv(url)


# In[23]:


df.head()


# In[24]:


df.tail()


# In[25]:


df.shape


# In[26]:


df.info() #number of rows and coloumns the dataset has


# In[27]:


df.describe()


# In[28]:


df.isnull().sum()


# We do not have any null values in our data set. So we can process further.

# # Visualizing the dataset

# We plot the dataset to check whether there is any relation between the two variables or not
# 
# 

# In[29]:


df.plot(x= 'Hours', y= 'Scores', style='+',color='red')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[30]:


df.corr() #corelation of the variables in the dataset


# We can clearly see here that there is a positive linear relationship between Hours and Scores which implies that if the number of hours increase the score will also increase. So we will use linear regression model for the predictions.

# # Making Predictions

# In[37]:


sns.distplot(df["Hours"],color ='green')


# In[38]:


sns.distplot(df["Scores"], color='red')


# After plotting the distribution plot of the two variables we can see that the variables are in a particular range and there are no outliers in the variable.

# In[64]:


import seaborn as sns
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df= pd.read_csv(url)
sns.countplot(x='Hours',data= df)




# In[11]:


sns.heatmap(df.corr(),cbar = True,linewidths = 0.5)


# In[123]:


sns.boxplot(x='Hours', y='Scores',data = df,width = 1.5,saturation=2,color='red',notch = True)


# # Preparing the data

# In[134]:


#using iloc function we divide data
x= df.iloc[:, :1].values
y= df.iloc[:, 1:].values


# In[129]:


x


# In[137]:


y


# # Modeling the data

# In[135]:


#splitting the data into train & test dataset
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)


# In[136]:


#training the model
from sklearn.linear_model import LinearRegression
model= LinearRegression()
model.fit(x_train,y_train)


# # Visualizing the model

# We are going to plot the best fit line here

# In[137]:


m=model.coef_
c=model.intercept_
line= m*x+c
#plotting for the training data
plt.scatter(x_train,y_train,color='green')
plt.plot(x,line,color = 'orange');
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# We will now fit this line in our test dataset

# In[141]:


plt.scatter(x_test,y_test,color='green')
plt.plot(x,line,color = 'orange');
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# # Model Prediction

# In[140]:


print(x_test)#testing data
y_pred = model.predict(x_test)#score prediction


# In[139]:


y_pred


# In[144]:


y_test


# In[145]:


#comparing actual vs predicted values
actual_vs_predicted= pd.DataFrame({ 'Actual':[y_test],'predicted':[y_pred]})
actual_vs_predicted


# # Predicted score if a student studies for 9.25 hours/day

# In[146]:


hours= 9.25
own_pred = model.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# # Model Evaluation:

# In[147]:


#evaluating the trained model
from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))


# #             THANK YOU!!!!         
