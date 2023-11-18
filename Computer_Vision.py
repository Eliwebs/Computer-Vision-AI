import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import PIL
from PIL import Image

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


# TESTING BASED ON OUR OWN HANDWRITTEN NUMBERS

def handwritten(image_path):
    # using PIL to open PNG image from local machine 
    test_img = Image.open(image_path)
    # converting picture to greyscale
    test_img = test_img.convert('L')
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


image_path = '/Users/EliWebster/Downloads/IMG_7378.jpg'


test_case = handwritten(image_path)

# This gives us the array of raw unnormalized scores for each number. The highest score is what the NN will predict.
prediction = NNmodel.predict(test_case)

# This gives us the highest score as a single digit from the unnormalized score. argmax takes the highest value from our array.
predicted_num = np.argmax(prediction, axis=1)

# Since predicted_num is an array with one element (the predicted class), we print this value using predicted_class[0]
print(f"\nPredicted digit:", predicted_num[0])

test_num = Image.open(image_path)
test_num.show()

# It sucks at identifying my images...


# quick for loop to test more than 1 number
img_start = 7378
for i in range(0, 9):
    img_start += 1
    img_start = str(img_start)
    image_path = '/Users/EliWebster/Downloads/IMG_' + img_start + '.jpg'
    print(image_path)

    test = handwritten(image_path)

    # This gives us the array of raw unnormalized scores for each number. The highest score is what the NN will predict.
    pred = NNmodel.predict(test)
    print(pred)

    # This gives us the highest score as a single digit from the unnormalized score. argmax takes the highest value from our array.
    predicted_nums = np.argmax(pred, axis=1)

# Since predicted_num is an array with one element (the predicted class), we can print this value using predicted_class[0]
    print(f"\nPredicted digitxxx:", predicted_nums[0])

    img_start = int(img_start)

# it still sucks