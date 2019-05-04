import numpy as np 

def svd(matrix):
  m, n = matrix.shape   
  print(m,n)
  mat1 = matrix.T.dot(matrix)
  mat2 = matrix.dot(matrix.T)
  print("m1:\n",mat1)
  print("m2:\n",mat2)
  
  sig1, v = np.linalg.eig(mat1)
  sig2, u = np.linalg.eig(mat2)
  print("sig1:\n",sig1,"v:\n", v)
  print("sig2:\n",sig2,"u:\n", u)

if __name__ == "__main__":
  A = np.array([[1,2,3],[3,4,5]])
  svd(A)
