# Pandas Cheat Sheet

Pandas is a powerful open-source data analysis and manipulation library for Python. It provides data structures like Series (1D) and DataFrame (2D) for efficient data handling.

## 1. Series Creation
- `pd.Series([1, 2, 3])`: Create a Series from a list.
- `pd.Series({'a': 1, 'b': 2})`: Create a Series from a dictionary.

## 2. DataFrame Creation
- `pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})`: From a dictionary of lists.
- `pd.read_csv('file.csv')`: Read from a CSV file.
- `pd.read_excel('file.xlsx')`: Read from an Excel file.
- `pd.read_sql('SELECT * FROM table', conn)`: Read from a SQL database.

## 3. Viewing Data
- `df.head(n)`: First `n` rows.
- `df.tail(n)`: Last `n` rows.
- `df.info()`: Summary of DataFrame, including data types and non-null values.
- `df.describe()`: Descriptive statistics for numerical columns.
- `df.shape`: Tuple of (rows, columns).
- `df.columns`: List of column names.
- `df.index`: Index of the DataFrame.

## 4. Selection and Indexing
- `df['col_name']`: Select a single column (returns a Series).
- `df[['col1', 'col2']]`: Select multiple columns (returns a DataFrame).
- `df.loc[row_label, col_label]`: Label-based indexing.
- `df.iloc[row_index, col_index]`: Integer-location based indexing.
- `df[df['col'] > 5]`: Boolean indexing for filtering rows.

## 5. Handling Missing Data
- `df.isnull()`: Boolean DataFrame indicating missing values.
- `df.notnull()`: Opposite of `isnull()`.
- `df.dropna()`: Drop rows with any missing values.
- `df.fillna(value)`: Fill missing values with a specified value.

## 6. Data Manipulation
- `df['new_col'] = df['col1'] + df['col2']`: Create a new column.
- `df.drop('col_name', axis=1)`: Drop a column.
- `df.drop(index=0)`: Drop a row by index.
- `df.groupby('col_name')`: Group data by a column.
- `df.sort_values(by='col_name')`: Sort by column values.
- `df.merge(df2, on='key_col')`: Merge DataFrames.
- `df.concat([df1, df2])`: Concatenate DataFrames.
- `df.apply(func)`: Apply a function along an axis.
- `df.pivot_table()`: Create a pivot table.

## 7. Unique Values and Counts
- `df['col'].unique()`: Get unique values in a column.
- `df['col'].nunique()`: Get number of unique values.
- `df['col'].value_counts()`: Get counts of unique values.

## 8. Data Cleaning
- `df.duplicated()`: Identify duplicate rows.
- `df.drop_duplicates()`: Remove duplicate rows.
- `df['col'].astype(new_type)`: Change data type of a column.

## 9. Input/Output
- `df.to_csv('output.csv', index=False)`: Write to CSV.
- `df.to_excel('output.xlsx', index=False)`: Write to Excel.
- `df.to_sql('table_name', conn, if_exists='replace')`: Write to SQL database.
