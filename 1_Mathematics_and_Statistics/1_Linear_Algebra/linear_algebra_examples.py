# Linear Algebra Examples using NumPy

import numpy as np

# 1. Scalars, Vectors, and Matrices
print("\n--- 1. Scalars, Vectors, and Matrices ---")
scalar = 5
print(f"Scalar: {scalar}")

vector = np.array([1, 2, 3])
print(f"Vector: {vector}")

matrix = np.array([[1, 2], [3, 4]])
print(f"Matrix:\n{matrix}")

# 2. Vector Operations
print("\n--- 2. Vector Operations ---")
v1 = np.array([1, 2])
v2 = np.array([3, 4])

# Addition
vec_add = v1 + v2
print(f"Vector Addition: {v1} + {v2} = {vec_add}")

# Scalar Multiplication
scalar_mult = 3 * v1
print(f"Scalar Multiplication: 3 * {v1} = {scalar_mult}")

# Dot Product
dot_product = np.dot(v1, v2)
print(f"Dot Product: {v1} . {v2} = {dot_product}")

# 3. Matrix Operations
print("\n--- 3. Matrix Operations ---")
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])

# Addition
mat_add = m1 + m2
print(f"Matrix Addition:\n{m1}\n+\n{m2}\n=\n{mat_add}")

# Matrix Multiplication
# For matrix multiplication, the number of columns in the first matrix must equal the number of rows in the second.
# Here, (2x2) * (2x2) = (2x2)
mat_mult = np.dot(m1, m2)
print(f"Matrix Multiplication:\n{m1}\n*\n{m2}\n=\n{mat_mult}")

# Transpose
mat_transpose = m1.T
print(f"Matrix Transpose of\n{m1}\n=\n{mat_transpose}")

# 4. Inverse Matrix
print("\n--- 4. Inverse Matrix ---")
# A singular matrix (determinant is 0) does not have an inverse.
# Let's use a non-singular matrix
inv_matrix = np.linalg.inv(m1)
print(f"Inverse of\n{m1}\n=\n{inv_matrix}")

# Verify: M * M_inv should be close to Identity Matrix
identity_check = np.dot(m1, inv_matrix)
print(f"M * M_inv (should be Identity):\n{identity_check}")

# 5. Determinant
print("\n--- 5. Determinant ---")
det_m1 = np.linalg.det(m1)
print(f"Determinant of\n{m1}\n= {det_m1}")

# 6. Eigenvalues and Eigenvectors
print("\n--- 6. Eigenvalues and Eigenvectors ---")
# For a simple 2x2 matrix
matrix_eigen = np.array([[2, 1], [1, 2]])
eigenvalues, eigenvectors = np.linalg.eig(matrix_eigen)

print(f"Matrix for Eigen-decomposition:\n{matrix_eigen}")
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verify: A * v = lambda * v
# For the first eigenvalue and eigenvector
v = eigenvectors[:, 0] # First eigenvector
lambda_val = eigenvalues[0] # First eigenvalue

Av = np.dot(matrix_eigen, v)
lambda_v = lambda_val * v

print(f"\nVerification for first eigenpair:")
print(f"A * v: {Av}")
print(f"lambda * v: {lambda_v}")
print(f"Are they close? {np.allclose(Av, lambda_v)}")
