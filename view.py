from keras.models import load_model
from keras.utils import plot_model
#from keras.layers.extra import TimeDistributedMaxPooling2D
model = load_model('model.json') #, custom_objects={'TimeDistributedMaxPooling2D':TimeDistributedMaxPooling2D})
plot_model(model, to_file='model.png')