import tensorflow as tf

model = tf.keras.Sequential()

#Add an input layer that will expect grayscale input images of size 256x256:
model.add(tf.keras.Input(shape=(256,256,1)))
#model.add(...)
model.add(tf.keras.layers.Flatten())
#Use a Flatten() layer to flatten the image into a single vector:

#model.add(...)

model.add(tf.keras.layers.Dense(100,activation="relu"))
model.add(tf.keras.layers.Dense(50,activation="relu"))
model.add(tf.keras.layers.Dense(2,activation="softmax"))

#Print model information:
model.summary() 
