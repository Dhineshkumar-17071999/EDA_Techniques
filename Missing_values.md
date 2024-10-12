# Find Missing Values
  df.isnull().sum()

# Handle Missing Values

Handling missing values is a critical step in data preprocessing because missing values can lead to biased analysis, reduce model performance, or cause errors in algorithms. There are several strategies to handle missing values depending on the nature of your data and the amount of missing information.

### Common Methods to Handle Missing Values

**1. Remove Missing Data:**
  - **Drop Rows:** Remove rows with missing values.
  - **Drop Columns:** Remove columns that contain too many missing values.

**2. Imputation (Replacing Missing Values):**
  - **Replace with Mean/Median/Mode:** Fill missing values with a statistical measure.
  - **Forward/Backward Fill:** Use neighboring values (particularly useful for time series data).
  - **Imputation Using Algorithms:** Use models to predict missing values (e.g., KNN imputation).

**3. Mark Missing Values:**
  - Add an indicator (binary feature) showing whether the value was missing or not.

### 1. Remove Missing Data

**A. Drop Rows with Missing Values**
You can drop rows with missing values using the **.dropna()** method.

```sh
# Drop rows with any missing values
df_cleaned = df.dropna()

# Drop rows where 'Age' is missing
df_cleaned = df.dropna(subset=['Age'])
```

**B. Drop Columns with Missing Values**
If a column contains too many missing values, you may decide to drop the entire column.

```sh
# Drop columns with any missing values
df_cleaned = df.dropna(axis=1)

# Drop columns where more than 50% of the values are missing
df_cleaned = df.dropna(thresh=len(df) * 0.5, axis=1)
```

### 2. Imputation (Filling Missing Values)

**A. Fill with Mean, Median, or Mode**
You can fill missing values in a column with the mean, median, or mode depending on the nature of the data.
  - **Mean:** For numerical data that is normally distributed.
  - **Median:** For numerical data that is skewed.
  - **Mode:** For categorical data.

```sh
# Fill missing values with the mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fill missing values with the median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing values with the mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
```

**B. Forward/Backward Fill**
In cases where missing values occur in a sequence (e.g., time series), you can use forward or backward fill.

```sh
# Forward fill (propagate the previous value forward)
df['Age'] = df['Age'].fillna(method='ffill')

# Backward fill (propagate the next value backward)
df['Age'] = df['Age'].fillna(method='bfill')
```

**C. Impute Using Algorithms (KNN Imputation, etc.)**
Advanced imputation methods like KNN Imputation or using regression models to predict missing values based on other features.
  - **K-Nearest Neighbors (KNN) Imputation:** Replaces missing values with the mean of the k-nearest neighbors.

```sh
from sklearn.impute import KNNImputer

# Create KNN imputer (with 5 nearest neighbors)
imputer = KNNImputer(n_neighbors=5)
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

### 3. Mark Missing Values
Sometimes it is useful to keep track of which values were missing by creating an indicator column.

```sh
# Create a new column to mark where 'Age' was missing
df['Age_missing'] = df['Age'].isnull().astype(int)

# Now you can impute the missing values while keeping track of where they occurred
df['Age'] = df['Age'].fillna(df['Age'].mean())
```


### Considerations When Handling Missing Data

**1. Understand the Cause of Missing Data:**
  - **Missing Completely at Random (MCAR):** The missing values are completely random and don't depend on other variables.
  - **Missing at Random (MAR):** The missing values depend on other observed variables.
  - **Missing Not at Random (MNAR):** The missing values are related to the variable itself (e.g., people with higher income may be less likely to report it).

**2. Percentage of Missing Data:**
  - **Low Percentage:** If only a small percentage of values are missing (e.g., less than 5%), you can impute or drop rows without losing much information.
  - **High Percentage:** If a column has a large percentage of missing values (e.g., more than 50%), consider dropping the column or imputing with more sophisticated methods.


### What is Biased Analysis
**Biased analysis** happens when the results or conclusions of an analysis are unfair or incorrect because the data or methods used are not properly balanced. This means the analysis is influenced by certain factors, leading to **misleading** or **skewed results**.

**How Biased Analysis Happens:**

  - **1. Wrong Data:** If the data used in the analysis does not represent the entire population or situation, the results will be biased.
      - Example: If you're analyzing people's favorite fruits but only ask people at a banana stand, your results will be biased toward bananas.

  - **2. Ignoring Important Information:** If key information is left out, the analysis may be biased because it doesn't have all the facts.
      - Example: If you're trying to predict house prices but ignore location, the analysis will be biased because location is a critical factor.

  - **3. Errors in Data:** If the data used is incorrect or incomplete, it can lead to biased analysis.
      - Example: If a temperature sensor always reads 5 degrees too high, your weather analysis will be biased.

  **Why It’s a Problem:**
  A biased analysis gives the wrong conclusions. For example, if a biased analysis says a product is very popular based on a bad survey, you might make poor decisions like overproducing the product, wasting time and resources.

  **In Simple Terms:**
    - Biased analysis = Unfair or incorrect results.
    - It happens when the data used isn't good, isn't complete, or is handled in a way that gives misleading answers.
    - Solution: Use good, balanced data and methods to avoid bias and get accurate results.
