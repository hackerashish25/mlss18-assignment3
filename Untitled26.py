
# coding: utf-8

# In[115]:


get_ipython().magic('matplotlib inline')


import os
import time

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale, StandardScaler
from sklearn.svm import LinearSVC


# In[116]:


from skimage import io

X = []
y = []

for sub_dir in os.listdir('data'):
    if not os.path.isdir(os.path.join('data', sub_dir)):
        continue
        
    label = int(sub_dir[1:])
    for file in os.listdir(os.path.join('data', sub_dir)):
        filename = os.path.join('data', sub_dir, file)
        image = io.imread(filename)
        X.append(image)
        y.append(label)
        
X = np.array(X, dtype='float64')
y = np.array(y)

print("X.shape: {}, y.shape: {}".format(X.shape, y.shape))
        



# In[117]:


imgplot = plt.imshow(X[0],cmap="gray")
plt.show()


# In[118]:


X[0]


# In[119]:


am=[]
for i in range(0,400):
    a=X[i,:,:]
    a=a.ravel()
    am.append(a)


# In[120]:


X=np.array(am)
X.shape


# In[121]:


num_labels  = 5
X_sample = X[y <= num_labels]
y_sample = y[y <= num_labels]


# In[122]:


X_sample_scaled = scale(X_sample)


# In[123]:


pca = PCA(n_components=2)
X_sample_2d=pca.fit_transform(X_sample_scaled)
X_sample_2d.shape


# In[124]:


fig = plt.figure(figsize=(8, 5))
X_train
# Go through documentation of this method
plt.scatter(X_sample_2d[:,0], X_sample_2d[:,1], c=y_sample, cmap=plt.cm.get_cmap('nipy_spectral', num_labels)) 

cb = plt.colorbar()
loc = np.arange(1, num_labels+1)
cb.set_ticks(loc)


# In[125]:


print("Memory used by X: {:.2f}MB".format(X.nbytes / 1024 / 1024))


# In[126]:


X_train,  X_temp,  y_train,  y_temp  = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# In[127]:


scaler  = StandardScaler()
scaler = scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)


# In[128]:


clf = LinearSVC(C=.001)
get_ipython().magic('time clf = clf.fit(X_train_scaled, y_train)')

print("Training accuracy: {:.4f}, Val Accuracy: {:.4f}".format(clf.score(X_train_scaled, y_train), clf.score(X_val_scaled, y_val)))


# In[129]:


clf = LogisticRegression(C=.001)
get_ipython().magic('time clf = clf.fit(X_train_scaled, y_train)')

print("Training accuracy: {:.4f}, Val Accuracy: {:.4f}".format(clf.score(X_train_scaled, y_train), clf.score(X_val_scaled, y_val)))


# In[130]:


X_train_scaled.shape


# In[131]:


pca=PCA(n_components=10304)
X_train=pca.fit_transform(X_train)


# In[132]:


a=pca.explained_variance_ratio_.cumsum()


# In[133]:


for i in range(10304):
    if (a[i]>=0.99):
        print(i)
        break


# In[134]:


pca=PCA(n_components=236)
X_train=pca.fit_transform(X_train)


# In[135]:


X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

print("Dimensionality reduced to:", pca.n_components_)


# In[136]:


clf = LogisticRegression(C=.01)
get_ipython().magic('time clf = clf.fit(X_train_pca, y_train)')

print("Training accuracy: {:.4f}, Val Accuracy: {:.4f}".format(clf.score(X_train_pca, y_train), clf.score(X_val_pca, y_val)))


# In[137]:


# Couple of utilities functions
def plot_gallery(images, titles, rows=3, cols=4):
    plt.figure()
    for i in range(rows * cols):
        fig = plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())
    plt.tight_layout()
    plt.show()
        
def titles(y_pred, y_test):
    for i in range(y_pred.shape[0]):
        yield 'predicted: {0}\ntrue: {1}'.format(y_pred[i], y_test[i])


# In[138]:


X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)


# In[139]:


y_pred = clf.predict(X_test_pca)
print('Accuracy score: {:.4f}'.format(accuracy_score(y_test, y_pred)))


# In[140]:


prediction_titles = list(titles(y_pred, y_test))
plot_gallery(X_test.reshape(-1, 112, 92), prediction_titles)

