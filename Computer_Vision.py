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
NNmodel.compile(optimizer = 'Adam', 
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
                metrics=['accuracy'])

# Training model with 10 epochs
NNmodel.fit(train_flat, y_train, epochs = 10)



# MODEL TESTING

# Testing accuracy and loss on test data

loss, accuracy = NNmodel.evaluate(test_flat, y_test, verbose = 2)

print(f'\nTest loss', loss)
print(f'\nTest accuracy', accuracy)


# TESTING SINGLE HANDWRITTEN NUMBERS

def handwritten(image_path):
    # using PIL to open PNG image from local machine 
    test_img = Image.open(image_path)
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
    img_flat = img_array.reshape(-1)
    # add another dimension to np array so that TF can use image
    img_batch = np.expand_dims(img_flat, axis=0)

    # returning preprocessed image
    return img_batch


#image_path = '/Users/EliWebster/Downloads/5_handwritten.png'


#test_case = handwritten(image_path)



# This gives us the array of raw unnormalized scores for each number. The highest score is what the NN will predict.
#prediction = NNmodel.predict(test_case)
#print(prediction)

# This gives us the highest score as a single digit from the unnormalized score. argmax takes the highest value from our array.
#predicted_num = np.argmax(prediction, axis=1)

# Since predicted_num is an array with one element (the predicted class), we print this value using predicted_class[0]
#print(f"\nPredicted digit:", predicted_num[0])

#test_num = Image.open(image_path)
#test_num.show()

# It sucks at identifying single images...


# function for converting a file pulled from directory

def handwritten_folder(image_path):
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

# Initialize a list to store images
image_list = []

# Loop through all files in the folder
for filename in os.listdir(IMG_pathdirectory):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other extensions if needed
        file_path = os.path.join(IMG_pathdirectory, filename)
        
        # Open the image file
        img = Image.open(file_path)
        # Preprocess image
        img_prepro = handwritten_folder(img)
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

# This code below will print the shape of each image, should be 784 (it is)
#for i, img in enumerate(image_array):
#    print(f"Shape of image {i}: {img.shape}")

# Stack the flattened images so it is in format (x, 784) - same as MNIST data
array_flat = np.vstack(image_array)
# checking that shape is (x,784) (it is)
print(f"Shape of array_flat: {array_flat.shape}")

# use model.evaluate to test modoel's performance on testing data

loss1, accuracy1 = NNmodel.evaluate(array_flat, labels_array, verbose = 2)
print(f'\nTest loss', loss1)
print(f'\nTest accuracy', accuracy1)

# Plotting one preprocessed image to ensure visually similar to MNIST
plt.figure(figsize=(1,1))
plt.plot(1, 1)
image = image_list[0]
image = image.reshape(28,28)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()



# Plotting data - currently manual. Next step - pull accuracy and loss directly from model.evaluate and populate list 
# Plotting process will be automatic if this is done correctly

epochs = range(1, 11)
#training_accuracy = [0.9338, 0.9729, 0.9817, 0.9861, 0.9899, 0.9917, 0.9940, 0.9951, 0.9961, 0.9961]
#training_loss = [0.2249, 0.0914, 0.0613, 0.0442, 0.0322, 0.0261, 0.0188, 0.0163, 0.0125, 0.0117]

training_accuracy = [0.9346, 0.9717, 0.9805, 0.9864, 0.9892, 0.9913, 0.9936, 0.9948, 0.9949, 0.9963 ]  # Example data
training_loss = [0.2282, 0.0940, 0.0621, 0.0452, 0.0349, 0.0265, 0.0203, 0.0157, 0.0146, 0.0107]  # Example data

plt.figure(figsize=(12, 6))

# Plotting testing accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, training_accuracy,'bo-', label='Testing Accuracy')
plt.title('Testing Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plotting testing loss
plt.subplot(1, 2, 2)
plt.plot(epochs, training_loss, 'ro-', label='Testing Loss')
plt.title('Testing Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()



test_accuracy = 0.9782000184059143
test_loss = 0.08583325892686844

plt.figure(figsize=(8, 6))

# Creating bars
plt.bar(['Test Accuracy', 'Test Loss'], [test_accuracy, test_loss], color=['blue', 'red'])

# Adding title and labels
plt.title('Test Accuracy and Loss')
plt.ylabel('Value')

# Showing the plot
plt.show()
