# Python for Data Science Examples

## NumPy
NumPy (Numerical Python) is a fundamental package for numerical computation in Python. It provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

```python
import numpy as np

# 1. Array Creation
print("--- NumPy Array Creation ---")
arr1d = np.array([1, 2, 3, 4, 5])
print("1D Array:", arr1d)

arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array:\n", arr2d)

zeros_arr = np.zeros((3, 4))
print("Zeros Array (3x4):\n", zeros_arr)

ones_arr = np.ones((2, 2))
print("Ones Array (2x2):\n", ones_arr)

range_arr = np.arange(10)
print("Range Array (0-9):", range_arr)

linspace_arr = np.linspace(0, 1, 5) # 5 evenly spaced values between 0 and 1
print("Linspace Array (0 to 1, 5 values):", linspace_arr)

rand_arr = np.random.rand(2, 3)
print("Random Array (2x3):\n", rand_arr)

# 2. Array Attributes
print("\n--- NumPy Array Attributes ---")
print("Shape of arr2d:", arr2d.shape)
print("Dimensions of arr2d:", arr2d.ndim)
print("Size of arr2d:", arr2d.size)
print("Data type of arr2d:", arr2d.dtype)

# 3. Array Indexing and Slicing
print("\n--- NumPy Array Indexing and Slicing ---")
print("First element of arr1d:", arr1d[0])
print("Element at (0, 1) of arr2d:", arr2d[0, 1])
print("First two rows of arr2d:\n", arr2d[0:2])
print("Second column of arr2d:\n", arr2d[:, 1])
print("Elements of arr1d greater than 3:", arr1d[arr1d > 3])

# 4. Array Manipulation
print("\n--- NumPy Array Manipulation ---")
reshaped_arr = range_arr.reshape((2, 5))
print("Reshaped Array (2x5):\n", reshaped_arr)

flattened_arr = reshaped_arr.flatten()
print("Flattened Array:", flattened_arr)

arr_a = np.array([1, 2])
arr_b = np.array([3, 4])
concatenated_arr = np.concatenate((arr_a, arr_b))
print("Concatenated Array:", concatenated_arr)

# 5. Array Operations
print("\n--- NumPy Array Operations ---")
print("arr1d + 5:", arr1d + 5)
print("arr1d * 2:", arr1d * 2)

matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
print("Matrix Multiplication (matrix1 @ matrix2):\n", matrix1 @ matrix2)

print("Sum of arr1d:", np.sum(arr1d))
print("Mean of arr1d:", np.mean(arr1d))
print("Max of arr2d:", np.max(arr2d))
print("Standard Deviation of arr1d:", np.std(arr1d))

# 6. Broadcasting
print("\n--- NumPy Broadcasting ---")
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])
print("a + b (broadcasting):\n", a + b)
```

## Pandas
Pandas is a powerful open-source data analysis and manipulation library for Python. It provides data structures like Series (1D) and DataFrame (2D) for efficient data handling.

```python
import pandas as pd
import numpy as np

# 1. Series Creation
print("\n--- Pandas Series Creation ---")
s = pd.Series([10, 20, 30, 40, 50], name="MySeries")
print("Series:\n", s)

# 2. DataFrame Creation
print("\n--- Pandas DataFrame Creation ---")
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 22],
    'City': ['New York', 'London', 'Paris', 'New York', 'London'],
    'Salary': [60000, 80000, 75000, 62000, 58000]
}
df = pd.DataFrame(data)
print("DataFrame:\n", df)

# From a dictionary of Series
df_from_series = pd.DataFrame({'col1': pd.Series([1, 2]), 'col2': pd.Series([3, 4])})
print("\nDataFrame from Series:\n", df_from_series)

# 3. Viewing Data
print("\n--- Pandas Viewing Data ---")
print("Head (3 rows):\n", df.head(3))
print("Info:\n")
df.info()
print("Describe:\n", df.describe())
print("Shape:", df.shape)
print("Columns:", df.columns)

# 4. Selection and Indexing
print("\n--- Pandas Selection and Indexing ---")
print("Select 'Name' column:\n", df['Name'])
print("Select 'Name' and 'Age' columns:\n", df[['Name', 'Age']])
print("Using .loc (row 0, 'Name'):", df.loc[0, 'Name'])
print("Using .iloc (row 1, col 2):", df.iloc[1, 2])
print("Filter by Age > 30:\n", df[df['Age'] > 30])

# 5. Handling Missing Data
print("\n--- Pandas Handling Missing Data ---")
df_missing = df.copy()
df_missing.loc[1, 'Salary'] = np.nan
df_missing.loc[3, 'City'] = np.nan
print("DataFrame with missing values:\n", df_missing)
print("Missing values count:\n", df_missing.isnull().sum())

df_filled = df_missing.fillna({'Salary': df_missing['Salary'].mean(), 'City': 'Unknown'})
print("Filled missing values:\n", df_filled)

df_dropped = df_missing.dropna()
print("Dropped rows with missing values:\n", df_dropped)

# 6. Data Manipulation
print("\n--- Pandas Data Manipulation ---")
df['Age_Plus_5'] = df['Age'] + 5
print("New column 'Age_Plus_5':\n", df)

print("Group by 'City' and calculate mean Salary:\n", df.groupby('City')['Salary'].mean())

print("Sorted by Salary (descending):\n", df.sort_values(by='Salary', ascending=False))

# 7. Unique Values and Counts
print("\n--- Pandas Unique Values and Counts ---")
print("Unique Cities:", df['City'].unique())
print("Number of unique Cities:", df['City'].nunique())
print("Value counts for City:\n", df['City'].value_counts())

# 8. Data Cleaning
print("\n--- Pandas Data Cleaning ---")
df_duplicates = pd.DataFrame({'A': [1, 2, 2, 3], 'B': ['x', 'y', 'y', 'z']})
print("DataFrame with duplicates:\n", df_duplicates)
print("Duplicated rows:\n", df_duplicates.duplicated())
print("DataFrame after dropping duplicates:\n", df_duplicates.drop_duplicates())

df['Age'] = df['Age'].astype(float)
print("Age column after changing dtype to float:\n", df['Age'].dtype)
```