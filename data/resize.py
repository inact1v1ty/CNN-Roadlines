import pickle
import numpy as np
import scipy.misc
from PIL import Image

train_images = pickle.load(open("full_CNN_train.p", "rb" ))
train_labels = pickle.load(open("full_CNN_labels.p", "rb" ))

train_images = np.array(train_images)
train_labels = np.array(train_labels)

train_images_resized = []
train_labels_resized = []

c = 0
max = len(train_images) + len(train_labels)

for label in train_labels:
    label = np.array([[j[0] for j in i] for i in label])
    label = Image.fromarray(label, 'L')
    label = label.resize((160, 120), Image.ANTIALIAS)
    label = np.array(label)
    label = np.array([[[j] for j in i] for i in label])
    train_labels_resized.append(label)
    c += 1
    print('{0:.2f}%'.format(c * 100 / max))

for img in train_images:
    img = Image.fromarray(img)
    img = img.resize((160, 120), Image.ANTIALIAS)
    img = np.array(img)
    train_images_resized.append(img)
    c += 1
    print('{0:.2f}%'.format(c * 100 / max))

pickle.dump(train_images_resized, open("full_CNN_train_r_m.p", "wb" ))
pickle.dump(train_labels_resized, open("full_CNN_labels_r_m.p", "wb" ))