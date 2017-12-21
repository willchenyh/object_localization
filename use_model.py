from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.applications.xception import Xception, preprocess_input

import numpy as np

model = Xception(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print 'Predicted:', imagenet_utils.decode_predictions(preds, top=3)[0]

