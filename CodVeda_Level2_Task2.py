#!/usr/bin/env python
# coding: utf-8

# #  Decision Tree Classification on Iris Dataset
# 
# ##  Project Overview
# This project demonstrates how to build a **Decision Tree Classifier** to predict the species of flowers using the famous Iris dataset.
# 
# ---
# 
# ##  Objectives
# - Train a Decision Tree model on labeled data
# - Visualize the decision tree
# - Apply pruning to avoid overfitting
# - Evaluate model performance using metrics
# - Interpret feature importance
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


df = pd.read_csv("iris.csv")
df.head()


# In[3]:


df.info()
df.describe()


# ## Check Missing Values

# In[4]:


df.isnull().sum()


# ## EDA

# In[5]:


sns.pairplot(df, hue='species')
plt.show()


# ## Feature & Target Split

# In[6]:


X = df.drop('species', axis=1)
y = df['species']


# ## Train-Test Split

# In[7]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ## Train Basic Decision Tree

# In[8]:


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)


# ## Predictions & Accuracy

# In[9]:


y_pred = dt.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ## Confusion Matrix

# In[10]:


cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=y.unique(),
            yticklabels=y.unique())

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ## Decision Tree Visualization

# In[11]:


plt.figure(figsize=(15,10))
plot_tree(dt, feature_names=X.columns, class_names=y.unique(), filled=True)
plt.show()


# ## Feature Importance

# In[12]:


importance = pd.Series(dt.feature_importances_, index=X.columns)

importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()


# ## Hyperparameter Tuning

# In[13]:


param_grid = {
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                    param_grid,
                    cv=5)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)


# ## Train Optimized Model

# In[14]:


best_dt = grid.best_estimator_

y_pred_best = best_dt.predict(X_test)

print("Optimized Accuracy:", accuracy_score(y_test, y_pred_best))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))


# ## Pruned Tree Visualization

# In[15]:


plt.figure(figsize=(15,10))
plot_tree(best_dt, feature_names=X.columns, class_names=y.unique(), filled=True)
plt.show()


# ## Before vs After

# In[16]:


print("Before Tuning Accuracy:", accuracy_score(y_test, y_pred))
print("After Tuning Accuracy:", accuracy_score(y_test, y_pred_best))


# ##  Key Insights
# 
# - Decision Trees are highly interpretable models.
# - Feature importance shows which features influence predictions the most.
# - Pruning helps reduce overfitting and improves generalization.
# - Hyperparameter tuning significantly enhances performance.
# 
# ---
# 
# ##  Conclusion
# 
# This project successfully demonstrates:
# 1. Model training  
# 2. Visualization  
# 3. Pruning  
# 4. Evaluation  
# 

# In[ ]:




