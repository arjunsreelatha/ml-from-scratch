import numpy as np

# shape of matrix (rows, columns)
A = np.array([[1, 2, 3],
              [4, 5, 6]])

B = np.array([[7, 8],
              [9, 8],
              [11, 12]])

print("Shape of A:", A.shape)
print("Shape of B:", B.shape)

C = np.array([[1, 2],
              [7, 8]])

D = np.array([[4, 6],
              [7, 8]])

# matrix addition
print("\nC + D =\n", C + D)

# scalar multiplication
print("\n3 * C =\n", 3 * C)

# element-wise multiplication
print("\nC * D =\n", C * D)

# matrix multiplication using @
print("\nA @ B =\n", A @ B)

# matrix multiplication using dot()
print("\nnp.dot(C, D) =\n", np.dot(C, D))

# transpose using np.transpose()
print("\ntranspose(A) =\n", np.transpose(A))

# transpose using .T
print("\nA.T =\n", A.T)

# inverse
print("\nInverse of C =\n", np.linalg.inv(C))

# determinant
print("\nDeterminant of D =", np.linalg.det(D))