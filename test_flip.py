from keras.preprocessing import image
import cv2
import numpy as np

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)

x_hor = image.flip_axis(x, 1)
cv2.imwrite('ele_hor.jpg', x_hor)
print 'x hor,', x_hor.shape

x_ver = image.flip_axis(x, 0)
cv2.imwrite('ele_ver.jpg', x_ver)
print 'x ver,', x_ver.shape

x_horver = image.flip_axis(x_hor, 0)
cv2.imwrite('ele_horver.jpg', x_horver)
print 'x horver', x_horver.shape


print "=============================================="
x = np.expand_dims(x, axis=0)
print 'x ,', x_hor.shape

x_hor = image.flip_axis(x, 2)
cv2.imwrite('ele_hor.jpg', x_hor[0,:,:,:])
print 'x hor,', x_hor.shape

x_ver = image.flip_axis(x, 1)
cv2.imwrite('ele_ver.jpg', x_ver[0,:,:,:])
print 'x ver,', x_ver.shape

x_horver = image.flip_axis(x_hor, 1)
cv2.imwrite('ele_horver.jpg', x_horver[0,:,:,:])
print 'x horver', x_horver.shape
