import numpy as np

A=np.array([[1,2,3],
           [4,5,6]])
B = np.array([[7,8],
             [9,8],
             [11,12]])
print("Shape of A:", A.shape)
print("Shape of B:", B.shape)

C = np.array([[1,2],
              [7,8]])

D = np.array([[4,6],
              [7,8]])

print("\nC + D =\n", C+D)

print("\n3 * C= \n", 3*C)

print("\nC * D = \n", C*D)