import numpy as np

# pca,  S = X.dot(X.T) = U sigma^2 U
# target: find the maximum k eigenvector of S, S is symmetric matrix X.dot(X.T)
#    using svd, X = U sigma V, we can direct find U  
#
# data: shape (dimension_number=p, observation_number=n)
#    p: source dimension 
#    m: target dimension

def pca(data, p, m, thereshold=100):
  # zero mean
  data -= np.mean(data)
  p, n = data.shape # p is dimention, n is observation

  if n < thereshold: 
    # svd to find U
    u, s, v = np.linalg.svd(data)
  else:
    eigenvalues, eigenvectors = np.linalg.eig(data.dot(data.T)) 
    u = eigenvectors[::-1]   # [::-1] inverse the order 
    s = np.sqrt(eigenvalues[::-1])
    # print(u.shape)
    # print(s.shape)

  return u[:m], s[:m]


if __name__ == "__main__":
  x = np.random.rand(10,10)
  print(x.shape)
  u,s=pca(x, 10, 5)
  print(u.shape)
  print(s.shape)
