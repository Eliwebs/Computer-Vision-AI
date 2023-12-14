import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import PIL
from PIL import Image
import PIL.ImageOps
import os
import pathlib

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
plt.title('25 random MNIST Images')
for i in range(len(Randnums)):
    plt.subplot(5, 5, i + 1)
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
plt.show()

# End of PRASHANT BANERJEE code
#---------------------------------------------------------------------------------------------------------------------------



# PREPROCESSING
#---------------------------------------------------------------------------------------------------------------------------

# normalize pixel values between 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten data set to single dimension array of size 'counts' by total # of pixels 
# counts for train = 60000, for test = 10000, # of pixels is 28x28 = 784 per image
train_flat, test_flat = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)



# train_flat = (60000, 784)
# test_flat = (10000, 784)
print( '\n',train_flat.shape)
print('\n', test_flat.shape)

# END OF PREPROCESSING
#---------------------------------------------------------------------------------------------------------------------------



# FUNCTIONS AND MODEL CREATION
#---------------------------------------------------------------------------------------------------------------------------

# Activation function (could directly use keras library - tf.keras.layers.Dense, activation = sigmoid, relU, etc)
# relU activation function checks value from previous layer, if x > 0, that value is unchanched, if x < 0, value is set to 0 
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
NNmodel.compile(optimizer = 'Adam', 
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
                metrics=['accuracy'])

# Training model with 10 epochs
graph_data = NNmodel.fit(train_flat, y_train, epochs = 10)
graph_data



# MODEL TESTING

# Testing accuracy and loss on test data

loss, accuracy = NNmodel.evaluate(test_flat, y_test, verbose = 2)

print(f'\nTest loss', loss)
print(f'\nTest accuracy', accuracy)


# END OF FUNCTIONS AND MODEL CREATION
#---------------------------------------------------------------------------------------------------------------------------



# TESTING OUR OWN DATASET OF HANDWRITTEN NUMBERS
#---------------------------------------------------------------------------------------------------------------------------

# function for preprocessing a file pulled from local directory

def handwritten_prepro(image_path):
    # using PIL to open PNG image from local machine 
    test_img = image_path
    # converting picture to greyscale
    test_img = test_img.convert('L')
    # Inverting black & white to match training data
    test_img = PIL.ImageOps.invert(test_img)
    # resizing image to 28x28
    test_img = test_img.resize((28, 28))
    # converting 28x28 into np array
    img_array = np.array(test_img)
    # standardizing pixel values between 0 & 1
    img_array = img_array / 255
    # reshaping to flatten image
    img_flat = img_array.reshape(784)
    # returning that flat array
    return img_flat


# path for image files
IMG_pathdirectory = '/Users/EliWebster/Downloads/Painted_Images'


# Initialize a list to store our own images, sorted by name so labels correctly match image.
image_list = []
directory = os.listdir(IMG_pathdirectory)
sorted_file = sorted(directory)

# Loop through all files in the folder
for filename in sorted_file:
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other extensions if needed
        file_path = os.path.join(IMG_pathdirectory, filename)
        
        # Open the image file
        img = Image.open(file_path)
        # Preprocess image
        img_prepro = handwritten_prepro(img)
        # Append preprocessed image to image_list
        image_list.append(img_prepro)


# Path for labels (single values separated by line is the correct format)
Label_pathdirectory = '/Users/EliWebster/Downloads/Painted_Labels.txt'

with open(Label_pathdirectory, 'r') as file:
    labels = file.read().splitlines()  #  each line in the file is a label, split file by line for list of labels
labels_array = np.array(labels, dtype=int)


# Converting image_list to np array so TF can process (this is an array of arrays)
image_list = np.array(image_list)
# reshaping so we have 50 flattened image arrays
image_array = image_list.reshape(image_list.shape[0], -1)



# Stack the flattened images so it is in format (x, 784) - same as MNIST data
array_flat = np.vstack(image_array)
# checking that shape is (x,784) (it is)
print(f"Shape of array_flat: {array_flat.shape}")


# use model.evaluate to test modoel's performance on testing data
loss1, accuracy1 = NNmodel.evaluate(array_flat, labels_array, verbose = 2)
print(f'\nHandwritten Test loss', loss1)
print(f'\nHandwritten Test accuracy', accuracy1)

# END OF TESTING OUR OWN DATASET OF HANDWRITTEN NUMBERS
#---------------------------------------------------------------------------------------------------------------------------



# VISUALIZING DATA WITH MATPLOTLIB
#---------------------------------------------------------------------------------------------------------------------------

# Plotting performance

# Accuracy
plt.plot(graph_data.history['accuracy'])
plt.plot(accuracy1)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
# Loss
plt.plot(graph_data.history['loss'])
plt.plot(loss1)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# Plotting one preprocessed image to ensure visually similar to MNIST
plt.figure(figsize=(1,1))
plt.plot(1, 1)
image = image_list[0]
image = image.reshape(28,28)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.suptitle(f'Label is: {labels_array[0]}', fontsize=16)
plt.show()


# Bar chart representing accuracy and loss for testing our own handwritten images
plt.figure(figsize=(8, 6))
plt.bar(['Test Accuracy', 'Test Loss'], [accuracy1, loss1], color=['blue', 'red'])
plt.title('Test Accuracy and Loss')
plt.ylabel('Value')
plt.show()


# END OF CODE
#---------------------------------------------------------------------------------------------------------------------------