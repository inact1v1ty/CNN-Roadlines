import pickle
import numpy as np
from matplotlib import pyplot as plt

train_images = pickle.load(open("full_CNN_train_r.p", "rb" ))
#train_labels = pickle.load(open("full_CNN_labels_r.p", "rb" ))

train_images = np.array(train_images)
#train_labels = np.array(train_labels)


#train_label_imgs = [[[[0, j[0], 0] for j in i] for i in label] for label in train_labels]

#train_label_imgs = np.array(train_label_imgs)

#train_label_imgs = train_label_imgs / 255

# now the real code :) 
curr_pos = 0

def key_event(e):
    global curr_pos

    if e.key == "right":
        curr_pos = curr_pos + 1
    elif e.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % len(train_images)

    img.imshow(train_images[curr_pos])
    img.plot()
    fig.canvas.draw()

fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', key_event)
img = fig.add_subplot(111)
img.imshow(train_images[curr_pos])
plt.show()