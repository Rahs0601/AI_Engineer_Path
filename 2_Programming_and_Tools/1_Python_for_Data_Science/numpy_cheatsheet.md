# NumPy Cheat Sheet

NumPy (Numerical Python) is a fundamental package for numerical computation in Python. It provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

## 1. Array Creation
- `np.array([1, 2, 3])`: Create a 1D array.
- `np.array([[1, 2], [3, 4]])`: Create a 2D array.
- `np.zeros((2, 3))`: Create an array of zeros with shape (2, 3).
- `np.ones((2, 3))`: Create an array of ones.
- `np.empty((2, 3))`: Create an empty array.
- `np.arange(10)`: Create an array with a range of values (0 to 9).
- `np.linspace(0, 1, 5)`: Create an array of evenly spaced values (5 values between 0 and 1).
- `np.random.rand(2, 3)`: Create an array with random values between 0 and 1.
- `np.random.randint(0, 10, (2, 3))`: Create an array with random integers.

## 2. Array Attributes
- `arr.shape`: Tuple of array dimensions.
- `arr.ndim`: Number of array dimensions.
- `arr.size`: Total number of elements in the array.
- `arr.dtype`: Data type of the array elements.

## 3. Array Indexing and Slicing
- `arr[0]`: Access the first element (1D).
- `arr[0, 1]`: Access element at row 0, column 1 (2D).
- `arr[0:2]`: Slice rows from 0 to 1.
- `arr[:, 1]`: Select all rows, column 1.
- `arr[arr > 5]`: Boolean indexing.

## 4. Array Manipulation
- `arr.reshape((2, 3))`: Reshape the array.
- `arr.flatten()`: Flatten the array to 1D.
- `np.concatenate((arr1, arr2))`: Concatenate arrays.
- `np.vstack((arr1, arr2))`: Stack arrays vertically.
- `np.hstack((arr1, arr2))`: Stack arrays horizontally.
- `np.split(arr, 2)`: Split array into multiple sub-arrays.

## 5. Array Operations
- **Arithmetic Operations:** `+`, `-`, `*`, `/`, `**` (element-wise).
- `np.dot(arr1, arr2)` or `arr1 @ arr2`: Matrix multiplication.
- `np.sum(arr)`: Sum of all elements.
- `np.mean(arr)`: Mean of all elements.
- `np.max(arr)`: Maximum element.
- `np.min(arr)`: Minimum element.
- `np.std(arr)`: Standard deviation.
- `np.exp(arr)`: Exponential of elements.
- `np.log(arr)`: Natural logarithm of elements.
- `np.sqrt(arr)`: Square root of elements.

## 6. Broadcasting
- NumPy's ability to perform operations on arrays of different shapes.
  ```python
  a = np.array([1, 2, 3])
  b = 2
  print(a * b) # Output: [2 4 6]
  ```

## 7. Universal Functions (ufunc)
- Functions that operate element-wise on arrays.
- Examples: `np.add`, `np.subtract`, `np.multiply`, `np.divide`, `np.sin`, `np.cos`.

## 8. Saving and Loading Arrays
- `np.save('my_array.npy', arr)`: Save array to a binary file.
- `np.load('my_array.npy')`: Load array from a binary file.
- `np.savetxt('my_array.txt', arr)`: Save array to a text file.
- `np.loadtxt('my_array.txt')`: Load array from a text file.
