import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets import mnist


# x = image y = label
# train = training data 
# test = testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# number of train labels for reference
unique, counts = np.unique(y_train, return_counts=True)
print("Train labels: ", dict(zip(unique, counts)))

# number of test labels for reference
unique, counts = np.unique(y_test, return_counts=True)
print("\nTest labels: ", dict(zip(unique, counts)))

# grabbing 25 random images 
Randnums = np.random.randint(0, x_train.shape[0], size=25)
images = x_train[Randnums]
labels = y_train[Randnums]


# plotting 25 random images
plt.figure(figsize=(5,5))
for i in range(len(Randnums)):
    plt.subplot(5, 5, i + 1)
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
plt.show()

# normalize pixel values between 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0