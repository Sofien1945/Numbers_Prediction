""""Numbers Prediction
Date: 12.10.2021
Part of SIMPLEARN Machine learning course examples
Done By: Sofien Abidi"""

# Import Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

digits = load_digits()

# Dataset features
print("Image data shape: ", digits.data.shape)
print("Label data shape: ", digits.target.shape)
# Random numbers visualization
plt.figure(figsize=(20, 4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize=20)

# Split the date to test and train
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(digits.data, digits.target, test_size=0.23, random_state=2)
print(train_X.shape)
print(test_X.shape)
print(train_y.shape)
print(test_y.shape)

# Dataset features
print("Image data shape: ", digits.data.shape)
print("Label data shape: ", digits.target.shape)
# Random numbers visualization
plt.figure(figsize=(20, 4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize=20)

# Training and modeling
from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression()
logisticRegr.fit(train_X, train_y)

# Testing the model
y_pred = logisticRegr.predict(test_X)
score = logisticRegr.score(test_X, test_y)
print(score)

# Confusion Matrix visualization
cm = metrics.confusion_matrix(test_y, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = f'Accuracy Score: {score:.2f}'
plt.title(all_sample_title, size=12)
plt.show()