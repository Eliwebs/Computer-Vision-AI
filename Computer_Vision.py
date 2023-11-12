import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# The following code (Separated by --) is from PRASHANT BANERJEE posted to "Kaggle" titled "MNIST - Deep Neural Network with Keras"
#---------------------------------------------------------------------------------------------------------------------------------------------
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

# End of PRASHANT BANERJEE code
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
print( '\n',train_flat.shape)
print('\n', test_flat.shape)



# Function for generating random weights for training data - not currently being used, if we create our own optimizer it will be useful
def training_wt(x, y):
    empty = []
    x = x_train
    y = y_train
    for i in range(x * y):
        empty.append(np.random.randn())
    return(np.array(empty).reshape(x, y))



# FUNCTIONS AND MODEL CREATION

# Activation function (could directly use keras library - tf.keras.layers.Dense, activation = sigmoid, relU, etc)
# relU activation function checks value from previous layer (all inputs are >= 0), if x > 0, that value is unchanched, if x < 0, value is set to 0 
def relUActivation(x):
    return(tf.maximum(0.0, x))

# Building model of NN
# Sequential groups a linear stack of layers into a keras model
# Dense gives hidden layers: activation(dot product of (input, weights) + bias) = output for next hidden layer
# Activation function is Relu shown above
# 256 layers are chosen because it provides better accuracy than 128, and about the same as >256.

NNmodel = tf.keras.Sequential([tf.keras.layers.Dense(256, activation = relUActivation), 
# Second Dense for final layer, condensing down to total classifiers possible (0-9 is 10 numbers)
                               tf.keras.layers.Dense(10)])


# Compile model using adam optimizer (one of the best for image classification), and SparseCategoricalCrossentropy as loss function 
# loss function SparseCC computes croossentropy loss between label and predictions.... 
# is used because our data (train_flat and test_flat) are flattened integer arrays
NNmodel.compile(optimizer = 'adam', 
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
                metrics=['accuracy'])

# Training model with 10 epochs
NNmodel.fit(train_flat, y_train, epochs = 10)



# MODEL TESTING

# Testing accuracy and loss on test data

loss, accuracy = NNmodel.evaluate(test_flat, y_test, verbose = 2)

print(f'\nTest loss', loss)
print(f'\nTest accuracy', accuracy)

