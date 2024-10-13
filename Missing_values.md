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

**When to Use Mean?:**
Use the mean when the data is normally distributed (symmetrical, bell-shaped). In a normal distribution, most values are close to the mean, and the mean gives a good summary of the data.

**When to Use Median?:**
Use the median when the data is skewed (not symmetrical). The median is less affected by outliers (extremely high or low values), making it a better measure of central tendency for skewed data.


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
  - A biased analysis gives the wrong conclusions.
  - For example, if a biased analysis says a product is very popular based on a bad survey, you might make poor decisions like overproducing the product, wasting time and resources.

  **In Simple Terms:**
  - Biased analysis = Unfair or incorrect results.
  - It happens when the data used isn't good, isn't complete, or is handled in a way that gives misleading answers.
  - Solution: Use good, balanced data and methods to avoid bias and get accurate results.


### Bias Simple explanation:
Bias is when something is unfairly tilted or influenced in a way that leads to incorrect or misleading results. It happens when certain factors are given more weight than they should, or some data is left out or misrepresented. Bias can distort the truth and make results unreliable.

**Everyday Example of Bias:**
Imagine you're doing a survey on people's favorite ice cream flavor, but you only ask people who are leaving a chocolate ice cream shop. Most of them will likely say "chocolate," but that doesn’t represent everyone’s true favorite. This is an example of bias in how you selected your participants (sample).

**Types of Bias:**
  - Data Bias: When data doesn't represent the whole population (e.g., only asking certain groups in a survey).
  - Measurement Bias: When tools or methods used to collect data are flawed (e.g., a faulty thermometer always reads 5 degrees too high).
  - Confirmation Bias: When people or analysts focus only on information that supports their existing beliefs and ignore contradictory data.


## Advance Techniques to Handle Missing Values
**Advanced Techniques for Imputation: Grouping by Other Factors**

When working with missing values, a more sophisticated approach involves using contextual information from other columns in the dataset to make imputation more accurate. This technique is called conditional or group-based imputation, and it allows you to use other variables to guide the imputation process.



Instead of simply filling the missing values with a single global mean or median, you can calculate the mean or median within groups of data that are similar in some way. This approach leverages relationships between features (columns) and often leads to better estimates.


Here are some common advanced imputation strategies:

**1. Group-Based Imputation**

This technique involves grouping the data by one or more categorical columns, and then filling in missing values based on the statistics of those groups (like the mean or median of each group).

**Example: Titanic Dataset (Pclass, Sex)** : The age column had some missing values

In the Titanic dataset, you could group passengers by class (Pclass) and gender (Sex). People from the same class and gender might have similar ages, so replacing missing values based on those groups provides more contextually accurate results.

```sh
# Group by 'Pclass' and 'Sex' and fill missing 'Age' values with the mean of each group
copy_train_df['Age'] = copy_train_df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))
```

  - **Explanation:** This code groups the DataFrame by Pclass and Sex, then fills missing values in the Age column with the mean age of each respective group.
  - **Why it's better:** A 1st-class male might have a different average age compared to a 3rd-class female. By imputing based on group means, you make a more informed guess about missing ages.


**Group by Multiple Columns:**

You can also group by multiple columns that might have an influence on the target column (in this case, Age), such as Embarked, Fare, or other categorical features in the dataset.

```sh
# Group by 'Pclass', 'Sex', and 'Embarked' to impute missing 'Age' values
copy_train_df['Age'] = copy_train_df.groupby(['Pclass', 'Sex', 'Embarked'])['Age'].transform(lambda x: x.fillna(x.mean()))
```

  - **Explanation:** This approach increases the specificity of the groups by adding the Embarked column, so you impute based on more detailed groupings.


## 2. Using Regression Models for Imputation

Instead of filling missing values with simple statistics (mean/median), you can use a regression model to predict the missing values based on other columns in the dataset.

**Example:**

You could train a regression model (like linear regression) where the target is the missing column (Age), and the features are other columns like Pclass, Fare, Sex, etc. Then, use the model to predict the missing values.


```sh
from sklearn.linear_model import LinearRegression

# Separate rows with missing and non-missing Age
train_data = copy_train_df[copy_train_df['Age'].notna()]
test_data = copy_train_df[copy_train_df['Age'].isna()]

# Features for the model (example: Pclass, Sex, Fare, Embarked)
X_train = train_data[['Pclass', 'Sex', 'Fare']]
y_train = train_data['Age']

X_test = test_data[['Pclass', 'Sex', 'Fare']]

# Convert categorical columns to numeric (e.g., using pd.get_dummies)
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict missing Age values
predicted_ages = model.predict(X_test)

# Fill missing Age values in the original dataframe
copy_train_df.loc[copy_train_df['Age'].isna(), 'Age'] = predicted_ages
```

  - **Explanation:** This code fits a linear regression model to predict the missing ages based on features like Pclass, Sex, and Fare. The model uses the patterns in the existing data (where Age is not missing) to predict the ages where it's missing.
  - **Why it's better:** Regression models can capture complex relationships between variables and provide more accurate estimates for missing values.


### 3. K-Nearest Neighbors (KNN) Imputation

The KNN imputation method looks for the most similar rows (neighbors) based on other features and fills in the missing value based on the values of those neighbors.


Using KNN, you look for rows with similar values for features like Pclass, Fare, Sex, and predict the missing value by averaging the Age values of the nearest neighbors.

```sh
from sklearn.impute import KNNImputer

# Features to use for KNN Imputation
knn_features = copy_train_df[['Pclass', 'Sex', 'Fare', 'Age']]

# Convert categorical features to numeric (e.g., Sex)
knn_features = pd.get_dummies(knn_features, drop_first=True)

# Initialize KNN Imputer
imputer = KNNImputer(n_neighbors=5)

# Perform KNN Imputation
copy_train_df['Age'] = imputer.fit_transform(knn_features)[:, -1]
```

  - **Explanation:** This approach uses the KNN algorithm to fill missing Age values. It finds the closest rows (based on other features) and imputes missing values by averaging the ages of the nearest neighbors.
  - **Why it's better:** KNN imputation is non-parametric, meaning it doesn't assume a particular distribution for the data. It can provide accurate imputation in complex datasets.


### 4. Using Machine Learning Imputation Libraries (like IterativeImputer)

The IterativeImputer from scikit-learn is a powerful tool that uses multivariate imputation. It fits a model iteratively, predicting missing values for each feature using all the other features.


```sh
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Select features for imputation
features = copy_train_df[['Pclass', 'Sex', 'Fare', 'Embarked', 'Age']]

# Convert categorical columns to numeric
features = pd.get_dummies(features, drop_first=True)

# Initialize Iterative Imputer
imputer = IterativeImputer()

# Perform imputation
imputed_data = imputer.fit_transform(features)

# Replace the Age column with imputed values
copy_train_df['Age'] = imputed_data[:, -1]
```


  - **Explanation:** The IterativeImputer works by using all the available features to iteratively predict the missing values, similar to regression but more robust.
  - **Why it's better:** This method accounts for the relationships between multiple features and provides a more sophisticated imputation method than simple group-based imputation.



## How to choose best method:

Choosing the best imputation method between Regression models, KNN imputation, and IterativeImputer depends on the dataset, the relationships between variables, and the overall goal of your analysis. Let's break down the strengths and weaknesses of each approach, followed by a recommendation for when to use each.

### 1. Regression Models (Linear/Other Regression Algorithms)

**Strengths:**

  - **Interpretable:** You can easily interpret how each feature contributes to the prediction of missing values.
  - **Fitted to your dataset:** You can train different regression models (linear, decision trees, random forests, etc.) to suit the structure of your data.
  - **Handles multiple features:** Regression can take into account several features to predict the missing value, and you can fine-tune the model by selecting only relevant features.
  - **Customizable:** You have full control over which model to use and can add more complex interactions between variables.

**Weaknesses:**

  - **Linear assumptions:** Basic linear regression assumes linear relationships between variables, which might not always hold.
  - **Feature preparation:** You need to handle categorical variables (e.g., encoding) and consider multicollinearity.
  - **Single imputation:** A regression model typically provides one predicted value per missing entry, which might not account for uncertainty in the imputation.

**Best Use Case:**

  - When the relationship between variables is well understood or known to be linear or easily captured by regression models.
  - Useful when you're dealing with small to medium-sized datasets and you want full control over the model used for imputation.
  - Ideal for imputation when you want an interpretable model or when missing values are relatively limited in scope.


### 2. KNN Imputation (K-Nearest Neighbors)

**Strengths:**

  - **Non-parametric:** KNN makes no assumptions about the distribution of the data or relationships between features, making it flexible for complex datasets.
  - **Captures local patterns:** It finds the closest observations (neighbors) and imputes missing values based on the local context, which can work well if similar rows exist in the dataset.
  - **Handles categorical and continuous variables:** Can handle both types of variables, provided they are preprocessed correctly.


**Weaknesses:**

  - **Computationally expensive:** KNN becomes slow with large datasets, as it needs to compute distances between each row and all others.
  - **Affected by noisy data:** If the dataset contains noisy observations or irrelevant features, the imputation might be less accurate.
  - **Ignores global patterns:** KNN focuses on local similarity but doesn't capture broader trends in the data.


**Best Use Case:**

  - Best for small to medium-sized datasets with few features, as KNN can be computationally expensive on larger datasets.
  - Works well when missing values occur randomly across observations, and when nearby data points (neighbors) are expected to have similar values for the missing feature.
  - Ideal for datasets with non-linear relationships and where assumptions about global patterns (like in regression) don't hold.


### 3. IterativeImputer (Multivariate Imputation)

**Strengths:**

  - **Multivariate imputation:** IterativeImputer uses all available features to predict missing values by iterating through each feature and predicting its missing values based on all others.
  - **Can handle complex relationships:** It doesn't rely on simple pairwise relationships and can capture more complex dependencies between multiple features.
  - **Multiple imputation models:** Each feature's missing values can be imputed using a different model (e.g., linear regression, decision trees), and the process iterates to refine the predictions.
  - **Handles large datasets well:** While more complex, it’s typically more scalable than KNN and can handle datasets with many features and observations.


**Weaknesses:**

  - **Complexity:** IterativeImputer is more complex to implement and understand compared to simpler methods like mean, median, or even regression imputation.
  - **Computational cost:** It can take time to converge, especially with large datasets or many missing values.
  - **Imputation variability:** Depending on the number of iterations and model used, there could be slight variations in imputed values (though it tends to be robust).


**Best Use Case:**

  - Ideal for large datasets with many missing values and multiple features, where the relationships between features are complex and not necessarily linear.
  - Useful when you need to account for multiple interactions between variables or want to avoid strong assumptions about the underlying data distribution.
  - Great for datasets where both categorical and continuous features are present, and the relationships between variables are intricate.


## Which One is Best?

  - If your dataset has strong linear relationships between variables or you need interpretability, go with a Regression Model. It's more customizable and suitable when you're certain about how variables relate to each other.
  - If your dataset has complex, non-linear relationships and is not too large, KNN Imputation can be a good choice because it captures local patterns without making assumptions about the overall distribution of the data. However, it may not scale well for larger datasets due to computational cost.
  - If you have a complex, large dataset with multiple missing values and intricate relationships, IterativeImputer is generally the most robust choice. It provides flexibility, uses multivariate approaches, and iterates to improve imputation accuracy. It’s less affected by specific assumptions about the distribution or relationships, making it ideal for complex, large datasets.



Each method has its strengths depending on your data characteristics, and sometimes it helps to try multiple methods and compare their performance using cross-validation or accuracy metrics from downstream tasks.
