import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib.pyplot
from tensorflow.keras import datasets, layers, models
import sys
import os
import io

# Force UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Optional: Suppress TensorFlow verbose logs (reduce noise)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"



(training_images, training_labels),(testing_images, testing_labels) = datasets.cifar10.load_data() # load training and testing data in the tuples 
training_images= training_images/255
testing_images= testing_images/255  #we can double assign at the same time like training_images, testing_images = training_images /255 , testing_images/255


#loading dataset from tensorflow keras . This is a dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 10 categories. See more info at the CIFAR homepage.

class_name = ['Plane','Car', 'Bird', 'Cat' ,'Deer','Dog','Frog','Horse','Ship', 'truck']

for i in range (16):
    plt.subplot(4,4,i+1) # Use plt.subplot as intended
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_name[training_labels[i][0]])


plt.show()
training_images=training_images[:50000]
training_labels=training_labels[:50000]
testing_images=testing_images[:10000]
testing_labels=testing_labels[:10000]

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images,testing_labels))

loss, accuracy =model.evaluate(testing_images,testing_labels)
print(f"Loss:{loss}")
print(f"Accuracy:{accuracy}")

model.save("img_classifier.model")
