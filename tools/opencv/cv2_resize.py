import cv2
import numpy as np
img = np.random.rand(210, 160 ,3)
print(img.shape)
img_scale = cv2.resize(img, (84, 84))
print(img_scale.shape)

