# Linear Algebra Concepts

Linear algebra is a branch of mathematics that deals with vector spaces and the linear mappings between them. It is fundamental to many areas of AI, especially in machine learning and deep learning, where data is often represented as vectors and matrices.

## Key Concepts:

### 1. Scalars, Vectors, and Matrices

*   **Scalar**: A single number.
    *   Example: `5`, `-3.14`
*   **Vector**: An array of numbers (a 1D array). Can represent a point in space or a direction.
    *   Example: `v = [1, 2, 3]` (a 3-dimensional vector)
*   **Matrix**: A 2D array of numbers. Used to represent data, transformations, and systems of equations.
    *   Example:
        ```
        A = [[1, 2],
             [3, 4]]
        ```

### 2. Vector Operations

*   **Addition/Subtraction**: Element-wise operation.
    *   `[a, b] + [c, d] = [a+c, b+d]`
*   **Scalar Multiplication**: Multiply each element by a scalar.
    *   `k * [a, b] = [k*a, k*b]`
*   **Dot Product (Inner Product)**: Multiplies corresponding elements and sums them. The result is a scalar.
    *   `[a, b] . [c, d] = a*c + b*d`
    *   Geometric interpretation: Related to the angle between vectors and projection.

### 3. Matrix Operations

*   **Addition/Subtraction**: Element-wise operation (matrices must have the same dimensions).
*   **Scalar Multiplication**: Multiply each element by a scalar.
*   **Matrix Multiplication**: More complex. If `A` is `m x n` and `B` is `n x p`, then `C = A * B` is `m x p`. The number of columns in `A` must equal the number of rows in `B`.
    *   Each element `C_ij` is the dot product of the i-th row of `A` and the j-th column of `B`.
*   **Transpose**: Swapping rows and columns. `A^T`.
    *   If `A = [[1, 2], [3, 4]]`, then `A^T = [[1, 3], [2, 4]]`

### 4. Special Matrices

*   **Identity Matrix (I)**: A square matrix with ones on the main diagonal and zeros elsewhere. Acts like the number 1 in multiplication (`A * I = A`).
*   **Inverse Matrix (A^-1)**: For a square matrix `A`, its inverse `A^-1` satisfies `A * A^-1 = I`. Used to "undo" a transformation or solve systems of linear equations.

### 5. Determinant

*   A scalar value that can be computed from the elements of a square matrix.
*   Indicates how much a linear transformation scales or flips space.
*   If `det(A) = 0`, the matrix is singular (non-invertible), meaning the transformation collapses space onto a lower dimension.

### 6. Eigenvalues and Eigenvectors

*   **Eigenvector**: A non-zero vector that, when a linear transformation is applied to it, only changes by a scalar factor (it only scales, doesn't change direction).
*   **Eigenvalue**: The scalar factor by which an eigenvector is scaled.
*   Equation: `A * v = λ * v`, where `A` is the matrix, `v` is the eigenvector, and `λ` (lambda) is the eigenvalue.
*   Importance: Used in dimensionality reduction (PCA), understanding stability of systems, and many other areas.

## Resources:

*   **Khan Academy**: Linear Algebra (online course)
*   **3Blue1Brown**: Essence of Linear Algebra (YouTube series)
*   **Textbook**: "Linear Algebra and Its Applications" by Gilbert Strang
