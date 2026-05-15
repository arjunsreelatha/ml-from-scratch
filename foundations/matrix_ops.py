import numpy as np

# Function to add two matrices
def add_matrices(A, B):
    return A + B

# Function to multiply a matrix by a scalar
def scalar_multiply(A, scalar):
    return scalar * A

# Function for element-wise multiplication
def elementwise_multiply(A, B):
    return A * B

# Function for matrix multiplication
def matrix_multiply(A, B):
    return A @ B

# Function to find transpose of a matrix
def transpose_matrix(A):
    return A.T

# Function to find inverse of a matrix
def inverse_matrix(A):
    # Inverse exists only for square matrices with non-zero determinant
    if A.shape[0] != A.shape[1]:
        return "Inverse not possible: matrix is not square"
    if np.linalg.det(A) == 0:
        return "Inverse not possible: determinant is zero"
    return np.linalg.inv(A)

# Function to find determinant of a matrix
def determinant_matrix(A):
    # Determinant exists only for square matrices
    if A.shape[0] != A.shape[1]:
        return "Determinant not possible: matrix is not square"
    return np.linalg.det(A)

# Main program
A = np.array([[1, 2, 3],
              [4, 5, 6]])

B = np.array([[7, 8],
              [9, 8],
              [11, 12]])

C = np.array([[1, 2],
              [7, 8]])

D = np.array([[4, 6],
              [7, 8]])

# shapes of matrices
print("Shape of A:", A.shape)
print("Shape of B:", B.shape)

# Matrix addition
print("\nC + D =\n", add_matrices(C, D))

# Scalar multiplication
print("\n3 * C =\n", scalar_multiply(C, 3))

# Element-wise multiplication
print("\nC * D =\n", elementwise_multiply(C, D))

# Matrix multiplication
print("\nA @ B =\n", matrix_multiply(A, B))

# Matrix multiplication using square matrices
print("\nC @ D =\n", matrix_multiply(C, D))

# Transpose
print("\nTranspose of A =\n", transpose_matrix(A))

# Inverse
print("\nInverse of C =\n", inverse_matrix(C))

# Determinant
print("\nDeterminant of D =\n", determinant_matrix(D))