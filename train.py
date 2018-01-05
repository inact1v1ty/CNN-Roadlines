import pickle
import numpy as np

from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.model_selection import train_test_split

# Load training images
train_images = pickle.load(open("full_CNN_train.p", "rb" ))

# Load image labels
labels = pickle.load(open("full_CNN_labels.p", "rb" ))

# Make into arrays as the neural network wants these
train_images = np.array(train_images)
labels = np.array(labels)

# Normalize labels - training images get normalized to start in the network
labels = labels / 255

#train_images = np.array(np.split(train_images, 1063))
#print(labels.shape)
#labels = np.array(np.split(labels, 1063))
#print(labels.shape)
#labels = np.array([i[2] for i in labels])
#print(labels.shape)

# Test size may be 10% or 20%
X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)

print('X: {0} -> Y: {1}'.format(X_train.shape, y_train.shape))

batch_size = 20
epochs = 100

# Using a generator to help the model use less data
# I found NOT using any image augmentation here surprisingly yielded slightly better results
# Channel shifts help with shadows but overall detection is worse
#datagen = ImageDataGenerator()
#datagen.fit(X_train)

model = load_model('model_d.json')

filepath="weights_d/weights.{epoch:02d}-{loss:.4f}-{acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, mode='max')

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,  
          write_graph=True, write_images=True)

model.summary()
#model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch = len(X_train),
#                   nb_epoch=epochs, verbose=1, validation_data=(X_val, y_val), callbacks=[checkpoint, tensorboard])
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint, tensorboard])

model.save_weights('weights_d.hdf5')

model.summary()