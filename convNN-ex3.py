# YOUR CODE SHOULD START HERE
import tensorflow as tf
import matplotlib.pyplot as plt
print(tf.__version__)
#callbacks
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc')>0.998):
            print("\nReached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()
#load_data
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
#this is to avoid an error of tf because my laptop
tf.logging.set_verbosity(tf.logging.ERROR)
##resize and normalize the dataset
training_images = training_images.reshape(60000,28,28,1)
training_images = training_images / 255.0
test_images =test_images.reshape(10000,28,28,1)
test_images = test_images / 255.0
#create and compile the ConvNN and DeepNN
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (28,28,1)),
 tf.keras.layers.MaxPooling2D(2,2), tf.keras.layers.Flatten(), tf.keras.layers.Dense(128, activation='relu'),
 tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
#to see a summary table of the NN
model.summary()
#Fit and evaluate the NN to asses the acc and loss in predictions
model.fit(training_images, training_labels, epochs = 5, callbacks=[callbacks])
test_loss = model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
plt.imshow(test_images[0])
plt.show()
#end
