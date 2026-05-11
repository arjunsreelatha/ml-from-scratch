import numpy as np

A=np.array([[1,2,3],
           [4,5,6]])
B = np.array([[7,8],
             [9,8],
             [11,12]])
#shape of matrix(rows,coloumn)
print("Shape of A:", A.shape)
print("Shape of B:", B.shape)

C = np.array([[1,2],
              [7,8]])

D = np.array([[4,6],
              [7,8]])
#matric addition
print("\nC + D =\n", C+D)
#scalar multiplication
print("\n3 * C= \n", 3*C)
#matrix element wise #multiplication
print("\nC * D = \n", C*D)
#matrix multiplication
print("\nA @ B = \n", A@B)