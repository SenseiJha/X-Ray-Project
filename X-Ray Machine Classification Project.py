import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy
DIRECTORY = "Covid19-dataset/train"
CLASS_MODE = "categorical"
COLOR_MODE = "grayscale"
TARGET_SIZE = (256,256)
BATCH_SIZE = 32

training_data_generator = ImageDataGenerator(
    rescale=1.0/255,
    zoom_range=0.1,
    rotation_range=25,
    width_shift_range=0.05,
    height_shift_range=0.05
)

training_iterator = training_data_generator.flow_from_directory(
    DIRECTORY,
    class_mode='categorical',
    color_mode='grayscale',
    batch_size=BATCH_SIZE
)

training_iterator.next()

print("Loading Validation Data")

validation_data_generator = ImageDataGenerator()

validation_iterator = validation_data_generator.flow_from_directory(
    DIRECTORY,
    class_mode='categorical', 
    color_mode='grayscale',
    batch_size=BATCH_SIZE
)

#Designing a Model
def design_model(training_data):
    # sequential model
    model = Sequential()

    # add input layer with grayscale image shape
    model.add(tf.keras.Input(shape=(256, 256, 1)))
    
    # convolutional hidden layers with relu functions
    model.add(layers.Conv2D(5, 5, strides=3, activation="relu"))
    model.add(layers.Conv2D(3, 3, strides=1, activation="relu"))
    
    # maxpooling layers and dropout layers as well
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(3,activation="softmax"))

    #Compile the Model
    print("Compiling Model")
    
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()]
    )
    model.summary()
    return model


#Creating a Model
model = design_model(training_iterator)

es = EarlyStopping(
    monitor='val_auc',
    mode='min',
    verbose=1,
    patience=20
)

#Training of Model
history = model.fit(
    training_iterator,
    steps_per_epoch=training_iterator.samples/BATCH_SIZE,
    epochs=5,
    validation_data=validation_iterator,
    validation_steps=validation_iterator.samples/BATCH_SIZE,
    callbacks=[es]
)

# plotting categorical and validation accuracy over epochs
fig = plt.figure()
ax1 = fig.add_subplot(4, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')

# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(4, 1, 3)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')
plt.savefig("Output.jpg")
plt.show()

#Predicted Outputs

test_steps_per_epoch = numpy.math.ceil(validation_iterator.samples / validation_iterator.batch_size)
predictions = model.predict(validation_iterator, steps=test_steps_per_epoch)

test_steps_per_epoch = numpy.math.ceil(validation_iterator.samples / validation_iterator.batch_size)
predicted_classes = numpy.argmax(predictions, axis=1)

true_classes = validation_iterator.classes
class_labels = list(validation_iterator.class_indices.keys())

report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

cm=confusion_matrix(true_classes,predicted_classes)
print(cm)
