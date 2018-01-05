import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, Deconv2D
from keras.layers import MaxPool2D, UpSampling2D
from keras.layers import Input, Concatenate
from keras.layers import Flatten, Reshape
from keras.layers.normalization import BatchNormalization

input_shape = (80, 160, 3)
pool_size = (2, 2)

### Model ###

u = Input(shape=input_shape)

# Normalizes incoming inputs. First layer needs the input shape to work
i = BatchNormalization(input_shape=input_shape)(u)

i = Conv2D(16, 3, padding='valid', activation='relu', name='Conv0')(i)

# A

a = Conv2D(32, 3, padding='valid', activation='relu', name='Conv1A')(i)

a = Dropout(0.1)(a)

a = Conv2D(32, 3, padding='valid', activation='relu', name='Conv2A')(a)

a = Dropout(0.1)(a)

a = MaxPool2D(pool_size=pool_size, name='Pooling1A')(a)

a = Conv2D(64, 3, padding='valid', activation='relu', name='Conv3A')(a)

a = Dropout(0.1)(a)

a = Conv2D(64, 3, padding='valid', activation='relu', name='Conv4A')(a)

a = Dropout(0.1)(a)

a = MaxPool2D(pool_size=pool_size, name='Pooling2A')(a)

b = Conv2D(32, 5, padding='valid', activation='relu', name='Conv1B')(i)

b = Dropout(0.1)(b)

b = MaxPool2D(pool_size=pool_size, name='Pooling1B')(b)

b = Conv2D(64, 5, padding='valid', activation='relu', name='Conv2B')(b)

b = Dropout(0.1)(b)

b = MaxPool2D(pool_size=pool_size, name='Pooling2B')(b)

g = Concatenate(axis=3)([a, b])

g = Conv2D(128, 3, padding='valid', activation='relu', name='Conv5')(g)

g = Dropout(0.1)(g)

g = Conv2D(128, 3, padding='valid', activation='relu', name='Conv6')(g)

g = Dropout(0.1)(g)

g = MaxPool2D(pool_size=pool_size, name='Pooling3')(g)

g = UpSampling2D(size=pool_size, name='UpSampling1A')(g)

g = Deconv2D(128, 3, padding='valid', activation='relu', name='Deconv1A')(g)

g = Dropout(0.1)(g)

g = Deconv2D(128, 3, padding='valid', activation='relu', name='Deconv2A')(g)

g = Dropout(0.1)(g)

g = UpSampling2D(size=pool_size, name='UpSampling2A')(g)

g = Deconv2D(64, 3, padding='valid', activation='relu', name='Deconv3A')(g)

g = Dropout(0.1)(g)

g = Deconv2D(64, 3, padding='valid', activation='relu', name='Deconv4A')(g)

g = Dropout(0.1)(g)

g = UpSampling2D(size=pool_size, name='UpSampling3A')(g)

g = Deconv2D(32, 3, padding='valid', activation='relu', name='Deconv5A')(g)

g = Dropout(0.1)(g)

g = Deconv2D(32, 3, padding='valid', activation='relu', name='Deconv6A')(g)

g = Dropout(0.1)(g)

#j = Concatenate(axis=3)([a, b])

j = Deconv2D(1, 5, padding='valid', activation='relu', name='Deconv0')(g)

# Model

model = Model(inputs=u, outputs=j)

### End of network ###
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

model.save('model_d.json')
