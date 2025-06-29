# Exploratory Data Analysis (EDA) Concepts

Exploratory Data Analysis (EDA) is a crucial step in the data science process. It involves analyzing data sets to summarize their main characteristics, often with visual methods. EDA is used to discover patterns, spot anomalies, test hypotheses, and check assumptions with the help of statistical graphics and other data visualization methods.

## Key Objectives of EDA:
- **Understand the data:** Get a feel for the dataset, its structure, and content.
- **Identify outliers and anomalies:** Detect unusual observations that might require further investigation.
- **Discover patterns and relationships:** Uncover correlations, trends, and groupings within the data.
- **Check assumptions:** Verify if the data meets the assumptions required for statistical modeling.
- **Prepare for modeling:** Clean, transform, and engineer features based on insights gained.

## Common Techniques and Tools:

### 1. Data Summarization:
- **Descriptive Statistics:** Mean, median, mode, standard deviation, variance, quartiles, etc.
- **`df.info()`:** Provides a concise summary of a DataFrame, including data types and non-null values.
- **`df.describe()`:** Generates descriptive statistics of numerical columns.
- **`df.value_counts()`:** Counts unique values in a Series.

### 2. Data Visualization:
- **Histograms:** Show the distribution of a single numerical variable.
- **Box Plots:** Display the distribution of numerical data and detect outliers.
- **Scatter Plots:** Visualize the relationship between two numerical variables.
- **Bar Charts:** Compare categorical data.
- **Heatmaps:** Show correlations between variables.
- **Pair Plots:** Visualize relationships between all pairs of numerical variables in a dataset.

### 3. Handling Missing Values:
- **Identification:** `df.isnull().sum()`
- **Imputation:** Filling missing values with mean, median, mode, or more advanced techniques.
- **Deletion:** Removing rows or columns with missing values.

### 4. Outlier Detection:
- **Z-score:** Measures how many standard deviations an element is from the mean.
- **IQR (Interquartile Range):** Used to define fences beyond which data points are considered outliers.
- **Visualization:** Box plots, scatter plots.

### 5. Feature Engineering (initial steps):
- **Creating new features:** Combining existing features or extracting information (e.g., from dates).
- **Transforming features:** Log transformation, standardization, normalization.

## Libraries for EDA in Python:
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical operations.
- **Matplotlib:** For basic plotting.
- **Seaborn:** For enhanced statistical data visualization.
- **Plotly/Bokeh:** For interactive visualizations.

## Example Workflow (Conceptual):
1. **Load Data:** `df = pd.read_csv('data.csv')`
2. **Initial Inspection:** `df.head()`, `df.info()`, `df.shape`
3. **Descriptive Statistics:** `df.describe()`
4. **Check Missing Values:** `df.isnull().sum()`
5. **Visualize Distributions:** Histograms for numerical, bar charts for categorical.
6. **Explore Relationships:** Scatter plots, heatmaps.
7. **Identify and Handle Outliers:** Using box plots and statistical methods.
8. **Consider Feature Engineering:** Based on insights.
