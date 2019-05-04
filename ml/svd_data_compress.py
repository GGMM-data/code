import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time


def svd_image_compress(path, compress_rate, w = 2, h = 3):
  image = Image.open(path)
  image_gray = image.convert('L')

  # show raw image
  plt.figure(1)
  plt.imshow(image)
  image = np.array(image)
  print(image.shape)

  # show gray image
  plt.figure(2)
  plt.imshow(image_gray)
  image_gray = np.array(image_gray)
  print(image_gray.shape)
  m, n = image_gray.shape
  
  # svd 
  u, sigma, v = np.linalg.svd(image_gray)

  # compute sum of the square singular
  square_singular_value_sum = np.square(sigma).sum()
  print(square_singular_value_sum)
  
  # compute the target singular 
  sum_target = compress_rate * square_singular_value_sum
  
  # find the kth singular
  begin_time = time.time()
  sum_k = 0
  for k in range(n):
    sum_k += np.square(sigma[k])
    if sum_k > sum_target:
      print(k)
      print(sum_k)
      break 
  end_time = time.time()
  print("time:", end_time - begin_time)
  
  # show w*h sub figure 
  figure,axes = plt.subplots(w, h, figsize=[40,20])
  axes = axes.flatten()
  
  new_sigma = np.zeros([m, n], dtype=np.float)
  for i in range(k + w * h): 
    new_sigma[i, i] = sigma[i] 
    new_image_gray = u.dot(new_sigma).dot(v)
    if i < k:
      continue
    axes[i-k].imshow(new_image_gray, cmap="gray")
    axes[i-k].set_title("%d th singular" % i)
    # plt.figure()
    # plt.imshow(new_image_gray, cmap="gray")
  
  plt.figure()
  if m >= n:
    img = u.dot(np.eye(m, n).dot(np.diag(sigma))).dot(v) 
  else:
    img = u.dot(np.diag(sigma).dot(np.eye(m, n))).dot(v) 
    
  plt.imshow(img, cmap="gray")

  plt.show()
  plt.close()


if __name__ == "__main__":
  path = "xh.jpg" 
  svd_image_compress(path, compress_rate=0.99)
