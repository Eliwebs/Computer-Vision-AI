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
    
#plt.show()


#---------------------------------------------------------------------------------------------------------------------------



# PREPROCESSING

# normalize pixel values between 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten data set to single dimension array of size 'counts' by total # of pixels 
# counts for train = 60000, for test = 10000, # of pixels is 28x28 = 784 per image
train_flat, test_flat = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)



# checking that works ^^^^^ (It does :)) 
# train_flat = (60000, 784)
# test_flat = (10000, 784)
print(train_flat.shape)
print('\n', test_flat.shape, '\n')



# Function for generating random weights for training data - not currently being used
def generate_training_wt(x, y):
    empty = []
    x = x_train
    y = y_train
    for i in range(x * y):
        empty.append(np.random.randn())
    return(np.array(empty).reshape(x, y))



#------------------------------------------------------------------------------------------------------------------------------



# Activation function (could directly use keras library - tf.keras.layers.Dense, activation = sigmoid, relU, etc)
def relUActivation(x):
    return(tf.maximum(0.0, x))

# Building model of NN
# Dense gives activation(dot product of (input, weights) + bias) = output for total layers
# 256 layers are chosen because it provides better accuracy than 128, and about the same as >256.

NNmodel = tf.keras.Sequential([tf.keras.layers.Dense(256, activation = relUActivation), 
# Second Dense for final layer, total classifiers possible (0-9 is 10 numbers)
                               tf.keras.layers.Dense(10)])


# Compile model using adam optimizer, and SparseCategoricalCrossentropy
# loss function Sparse.... is used because data is a flattened integer array, rather than one-hot
NNmodel.compile(optimizer = 'adam', 
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
                metrics=['accuracy'])

# Training model with 10 epochs
NNmodel.fit(train_flat, y_train, epochs = 10)


# Testing accuracy and loss on test data

loss, accuracy = NNmodel.evaluate(test_flat, y_test, verbose = 2)

print(f'\nTest loss', loss)
print(f'\nTest accuracy', accuracy)

