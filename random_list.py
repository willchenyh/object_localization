import numpy as np
import os
import random

file_name = 'random_list.txt'
src_dir = 'find_phone/'

f = open(file_name, 'wb')
# get list of image names
file_list = os.listdir(src_dir)
img_name_list = [fname+'\n' for fname in file_list if fname.endswith('.jpg')]
print len(img_name_list)
random.shuffle(img_name_list)

f.writelines(img_name_list)

f = open(file_name, 'rb')
img_name_list = f.readlines()
img_name_list = [img_name.strip() for img_name in img_name_list]
print len(img_name_list)
print img_name_list