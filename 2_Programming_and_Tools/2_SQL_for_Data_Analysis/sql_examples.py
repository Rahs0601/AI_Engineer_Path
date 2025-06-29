# SQL for Data Analysis Examples (using SQLite for simplicity)

import sqlite3
import pandas as pd

# Connect to an in-memory SQLite database
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# Create a table
cursor.execute('''
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    salary REAL
)
''')
conn.commit()

# Insert data
employees_data = [
    (1, 'Alice', 'HR', 60000),
    (2, 'Bob', 'Engineering', 80000),
    (3, 'Charlie', 'HR', 65000),
    (4, 'David', 'Engineering', 90000),
    (5, 'Eve', 'Marketing', 70000)
]
cursor.executemany("INSERT INTO employees (id, name, department, salary) VALUES (?, ?, ?, ?)", employees_data)
conn.commit()

print("--- SQL Query Examples ---")

# Example 1: Select all employees
print("\nAll Employees:")
cursor.execute("SELECT * FROM employees")
for row in cursor.fetchall():
    print(row)

# Example 2: Select employees from a specific department
print("\nEmployees in Engineering Department:")
cursor.execute("SELECT name, salary FROM employees WHERE department = 'Engineering'")
for row in cursor.fetchall():
    print(row)

# Example 3: Calculate average salary by department
print("\nAverage Salary by Department:")
cursor.execute("SELECT department, AVG(salary) FROM employees GROUP BY department")
for row in cursor.fetchall():
    print(row)

# Example 4: Order employees by salary
print("\nEmployees Ordered by Salary (Descending):")
cursor.execute("SELECT name, salary FROM employees ORDER BY salary DESC")
for row in cursor.fetchall():
    print(row)

# Using Pandas to read SQL query results
print("\nReading SQL query into Pandas DataFrame:")
df_employees = pd.read_sql_query("SELECT * FROM employees", conn)
print(df_employees)

# Close the connection
conn.close()
