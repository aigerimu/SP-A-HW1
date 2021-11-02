#!/usr/bin/env python
# coding: utf-8

# # Aigerim Gilmanova 
# 
# 

# #### Sensors and Sensing 
# #### Home  Work 1
# #### 29 October, 2021

# In[1]:


get_ipython().system('pip3 install pyransac3d')


# In[2]:


# Installing necessary linraries.
import pandas as pd
import seaborn as sb
import numpy as np
import math
import pyransac3d as pyrsc
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets


# # Task 1 (Case 2)

# In[3]:


# Reading the data from the desktop. It is necessary to write the path of the file. 
df = pd.read_table('C:\\Users\\Acer\\Desktop\\case2.txt') 


# In[4]:


# The data in provided file is in one column, so it is necessary to split them into two columns, 
# which are "Time" and "Roll Angle". 
# Then, data from string type was converted into float type.
df['Time'].str.split(',', expand=True)
df[['Time', 'Roll Angle']] = df['Time'].str.split(',', expand=True)
df[['Time', 'Roll Angle']] = df[['Time', 'Roll Angle']].astype(float)


# In[5]:


# Plotting data to observe outliers and its shape of distribution.
plt.plot(df['Time'], df['Roll Angle'], 'bo')
plt.title('Time vs Roll Angle')
plt.xlabel('Time (s)')
plt.ylabel('Roll Angle')
plt.show()


# In[6]:


# Finding the confidence interval by using formula of standart deviation, standart error and margin error. 
std = np.std(df['Roll Angle'])
std_df = std/math.sqrt(6700)
ME = std_df * 1.96 # 1.96 is the z-value of P(0.95+0.05/2)=0.975
mean = np.mean(df['Roll Angle'])
ci_from = mean - mean*ME/100 #upper bound
ci_to = mean + mean*ME/100 #lower bound
print('Confidence Interval is from ' + str(ci_from) + ' to ' + str(ci_to))


# In[7]:


# Using boxplot to see outliers.
sb.boxplot(data = df['Time'], x = df['Roll Angle'])
plt.title('Boxplot for outliers detection')


# In[8]:


# Removing outliers from the observations. 
df = df.drop(df[(df['Roll Angle'] > 14580)].index)
df = df.drop(df[(df['Roll Angle'] < 14400)].index)


# In[9]:


plt.plot(df['Time'], df['Roll Angle'], 'bo')
plt.title('Time vs Roll Angle')
plt.xlabel('Time (s)')
plt.ylabel('Roll Angle')
plt.show()


# In[10]:


# Using formula for linear Regression that is y = x.b, where x is a matrix of ones and x-values.
# And the formula of b is (x_transpose.x)^(-1).x_transpose.y, "." is a dot product.
x = df.values[:,0]
x = np.vstack(x)
ones = np.ones((len(x),1),dtype=float)
x = np.hstack((ones, x))
y = df.values[:,1]
y = np.vstack(y)
b = np.linalg.inv(np.transpose(x).dot(x)).dot(np.transpose(x).dot(y))
res = x.dot(b)


# In[11]:


# The Linear Regression model does not fit the data. 
plt.plot(x, y, 'bo')
plt.plot(x, res)
plt.title('Time vs Roll Angle')
plt.xlabel('Time (s)')
plt.ylabel('Roll Angle')
plt.show()


# # Task 2 (ID 08)

# In[12]:


# Reading the data from the desktop. It is necessary to write the path of the file. 
df1 = pd.read_table('C:\\Users\\Acer\\Desktop\\data_set_8_.txt')


# In[13]:


# The data in provided file is in one column, so it is necessary to split them into three columns, 
# which are "x", "y" and "z". 
# Then, data from string type was converted into float type.
df1['x'].str.split(',', expand=True)
df1[['x', 'y', 'z']] = df1['x'].str.split(',', expand=True)
df1[['x', 'y', 'z']] = df1[['x', 'y', 'z']].astype(float)


# In[14]:


# Converting data into np.array
x = df1.values[:,0]
x = np.vstack(x)
y = df1.values[:,1]
y = np.vstack(y)
z = df1.values[:,2]
z = np.vstack(z)


# In[15]:


# PLotting data in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)
plt.show()


# The following data points represent a plane. Also, it can be concluded before the plotting because there are three points (x,y,z) that is a plane. 

# In[16]:


# Using pyransac3d library to implement RANSAC. 
# To implement RANSAC function and fit the plane, it is necessary to include data points, threshold, min points 
# and the amount of maximum iterations. 
# Imported RANSAC function for plane returns the equation of a plane and inliers.  
points = df1.values
plane1 = pyrsc.Plane()
best_eq, best_inliers = plane1.fit(points, thresh=0.05, minPoints=3, maxIteration=1000)


# In[17]:


print ('The plane equation is ' + 'A*x + B*y + C*z + D')
print('The plane equation is' + ' ' + f'{best_eq[0]}' +'*x + ' +  f'{best_eq[1]}' +'*y + '+  f'{best_eq[2]}' +'*z ' +  f'{best_eq[3]}' )
#print(best_inliers)


# In[18]:


# Measuring z-values to plot the fitted plane.
A, B, C, D = best_eq
X,Y = np.meshgrid(x,y)
Z = (D - A*X - B*Y)/C


# In[19]:


# Plotting the plane.
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(X, Y, Z)
plt.show()


# The selected minimal sample set (MSS) is three because three points are necessary to model a plane. In the arguments of RANSAC build-in function, it is used as 'minPoints'. Usually, the optimal number of iterations are measured from the formula h =< log(delta)/log(1-q), where delta is a treshold and q is a probability of inliers. However, as we do not have a q, we used a graphical approach and a default argument of iterations of RANSAC build-in function, which is 1000. And the treshold level is selected by the users and depends on the nature of the problem, so here it is a default value, which is 0.05
