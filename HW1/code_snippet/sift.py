import cv2
from time import time
import numpy as np
features=[]

image = cv2.imread('./sift-scene.jpg', cv2.IMREAD_GRAYSCALE)

t1=time()
sift = cv2.xfeatures2d.SIFT_create()
kp, descriptors = sift.detectAndCompute(image, None)

for des in descriptors:
    features.append(des)

print('Compute time is {} '.format(time()-t1))

print('Shape of descriptors is {}'.format(np.shape(descriptors)))