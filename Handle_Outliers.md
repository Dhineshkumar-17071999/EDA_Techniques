# Handle Outliers

Handling outliers is a critical step in data preprocessing, as outliers can skew results, negatively impact machine learning model performance, or provide incorrect insights if not treated properly. The approach to handling outliers depends on the context of your dataset, the nature of the outliers, and whether or not they contain valuable information.

**Common Techniques to Handle Outliers**

Here are some widely-used methods to deal with outliers:

### 1. Remove Outliers (Trimming/Filtering)
One of the simplest approaches is to remove outliers from the dataset. This is effective when you are confident that the outliers are due to errors or irrelevant to the analysis. Be cautious with this method, as removing too much data can lead to loss of valuable information.

**Steps:**
  - Use **Z-score** or **IQR (Interquartile Range)** method to detect outliers.
  - Filter out rows where the feature values lie outside a specified threshold.

**Example in Python (Removing based on IQR):**

```sh
import pandas as pd

# Sample dataset
df = pd.DataFrame({'Fare': [10, 20, 30, 40, 500, 15, 25, 1000]})

# Calculate Q1 and Q3
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove rows that contain outliers
df_no_outliers = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]
print(df_no_outliers)
```

**When to Use:**
  - The outliers are a result of data entry errors or irrelevant information.
  - You have a large dataset, and removing a small number of outliers won’t affect the results significantly.

### 2. Imputation (Replacing Outliers)

Instead of removing outliers, you can replace them with more reasonable values, such as the mean, median, or mode of the data. This ensures you don’t lose any data, but it reduces the influence of extreme values.

**Steps:**
  - Identify outliers using Z-scores, IQR, or visual methods (box plot).
  - Replace outliers with mean/median/mode or another appropriate statistic.

**Example in Python (Replacing with Median):**

```sh
# Replace outliers in 'Fare' with the median
median_fare = df['Fare'].median()
df['Fare'] = df['Fare'].apply(lambda x: median_fare if x < lower_bound or x > upper_bound else x)
print(df)
```

**When to Use:**
  - You want to keep the size of the dataset unchanged but reduce the effect of extreme values.
  - Outliers are not errors but may represent rare values.

**When to use mean, meadian and mode:**
The choice between using mean, median, or mode depends on the characteristics of your data and the specific question you're trying to answer. These are measures of central tendency, and each has its strengths and weaknesses depending on the data distribution, the presence of outliers, and the type of data.

### 2.1 Mean (Arithmetic Average)
The mean is the sum of all values divided by the number of values. It’s the most commonly used measure of central tendency.

**When to Use the Mean:**
  - **Data is Symmetrically Distributed:** The mean is most useful when the data is normally distributed (bell-shaped curve) or when it has no significant outliers. In such cases, the mean accurately reflects the center of the data.
  - **Continuous Data:** The mean is best for continuous variables (e.g., height, weight, age).
  - **No Outliers:** The mean is sensitive to extreme values (outliers). If your dataset contains significant outliers, the mean can be misleading.


**Example:**
  - In a dataset of exam scores: [75, 80, 85, 90, 95], the mean would be 85.
  - Use the mean if the data is distributed evenly without skewness or extreme outliers.


### 2.2 Median (Middle Value)
The median is the middle value in a dataset when the values are arranged in ascending or descending order. If there is an even number of values, the median is the average of the two middle values.

**When to Use the Median:**
  - **Skewed Data:** The median is more appropriate when the data is skewed (i.e., it has a long tail on one side). For example, income data is often skewed because a few people earn very high incomes while most earn much less. In such cases, the median gives a better indication of the central tendency than the mean.
  - **Presence of Outliers:** The median is robust to outliers. It is not affected by extreme values, making it a good choice if your dataset contains significant outliers.
  - **Ordinal Data:** The median can be used for ordinal data (data that has a natural order but no consistent difference between values), such as satisfaction ratings (e.g., 1 = dissatisfied, 2 = neutral, 3 = satisfied).

**Example:**
  - In a dataset of house prices: [100,000, 150,000, 200,000, 2,000,000], the mean might be skewed by the expensive house, but the median (150,000) gives a more accurate reflection of typical house prices.
  - Use the median when the data contains extreme values or is not symmetrically distributed.


### 2.3. Mode (Most Frequent Value)
The mode is the value that appears most frequently in a dataset. There can be more than one mode in a dataset (if multiple values appear with the same frequency), or no mode at all if all values are unique.

**When to Use the Mode:**
  - **Categorical Data:** The mode is especially useful for categorical data where we want to know the most common category or class. For example, in survey data, the mode might tell you the most common response.
  - **Nominal Data:** When the data is nominal (categories without any order), such as favorite color or preferred brand, the mode is the best measure of central tendency.
  - **Bimodal/Multimodal Distributions:** If the dataset is bimodal (has two modes) or multimodal (more than two modes), the mode is useful for identifying the most frequent values.


**Example:**
  - In a dataset of favorite ice cream flavors: [Vanilla, Chocolate, Vanilla, Strawberry], the mode is Vanilla, as it appears most frequently.
  - Use the mode when you want to identify the most common category or value, especially in non-numeric data.


**When to Use Each Measure: Quick Overview**

| Measure | Best For | Situations to Use | Sensitivity to Outliers | Data Types |
| ------- | -------- | ----------------- | ----------------------- | ---------- |
| Mean | Normal/ Symmetric Data | - Data is symmetrically distributed -  No extreme outliers | High | Continuous |
| Median | Skewed/ Outlier-prone Data | - Data is skewed - Contains outliers - Ordinal data | Low (Resistant to outliers) | Continuous or Ordinal |
| Mode | Categorical/ Nominal Data | - Data is categorical (e.g., brand, color) - Need to find the most common value | Not applicable | Categorical or Ordinal |

**Examples in Action**
  - **Income Data:** Incomes are usually **skewed** because a few people earn extremely high salaries. The **median** income is more useful than the mean because the median is not affected by the very high incomes that would skew the mean.
  - **Exam Scores:** If most students scored around the same mark and there are no outliers, the **mean** would be appropriate to represent the average performance.
  - **Survey Data (Customer Satisfaction):** For a survey with responses like "Satisfied," "Neutral," and "Dissatisfied," the **mode** is useful because it tells you the most common response.
  - **House Prices:** House prices can have significant outliers (very expensive houses). In such cases, the **median** is often used to represent the central tendency because it’s less sensitive to extreme values than the mean.


### 3. Capping (Winsorization)

**Capping**, also called **winsorization**, is a method where extreme values (outliers) are capped at a certain percentile value. You don’t remove the outliers but **limit** them to a predefined threshold.

**Steps:**
  - Set a threshold at the upper and lower percentiles (e.g., 1st and 99th percentiles).
  - Set a threshold at the upper and lower percentiles (e.g., 1st and 99th percentiles).

**Example in Python (Capping):**

```sh
# Cap 'Fare' at 1st and 99th percentiles
lower_percentile = df['Fare'].quantile(0.01)
upper_percentile = df['Fare'].quantile(0.99)

df['Fare'] = df['Fare'].apply(lambda x: lower_percentile if x < lower_percentile else upper_percentile if x > upper_percentile else x)
print(df)
```

**When to Use:**
  - You want to reduce the effect of extreme values but not remove any data points.
  - Outliers are legitimate values but should be controlled to avoid skewing the analysis.

### 4. Transformation (Log, Square Root, or Box-Cox Transform)

Sometimes, data with outliers can be transformed to reduce the impact of the extreme values and bring the distribution closer to normal. Common transformations include:
  - **Log Transformation:** Works well when the data has positive skew.
  - **Square Root Transformation:** Helps stabilize variance in the data.
  - **Box-Cox Transformation:** Generalized transformation that can handle both positive and negative data.

**Example in Python (Log Transformation):**

```sh
import numpy as np

# Apply log transformation to the 'Fare' column
df['Fare_log'] = np.log(df['Fare'] + 1)  # Add 1 to avoid log(0) issues
print(df)
```

**When to Use:**
  - Your data is skewed, and you want to normalize the distribution.
  - The presence of outliers is making it difficult to fit the data to a normal distribution.

### 5. Binning

Binning is a technique that groups continuous data into bins or intervals. This method can help reduce the effect of outliers by placing extreme values into larger bins with other data points.

**Steps:**
  - Define bin edges based on your data range.
  - Group outliers into bins.

**Example in Python (Using pd.cut):**

```sh
# Bin 'Fare' into categories
df['Fare_bin'] = pd.cut(df['Fare'], bins=[0, 50, 100, 200, 1000], labels=['Low', 'Medium', 'High', 'Very High'])
print(df)
```

**When to Use:**
  - When you want to simplify the data by converting continuous values into categorical bins.
  - Outliers represent rare values that could still belong to a group but shouldn't be excluded.


### 6. Treating Outliers with Machine Learning Methods

**Isolation Forest:**
An unsupervised machine learning method that identifies outliers by randomly partitioning the data and checking how isolated a point is. Outliers are isolated quickly, while normal data points require more splits.

**Example in Python (Using Isolation Forest):**

```sh
from sklearn.ensemble import IsolationForest

# Fit isolation forest
iso = IsolationForest(contamination=0.1)  # Set contamination to the proportion of outliers expected
df['anomaly'] = iso.fit_predict(df[['Fare']])  # Use only relevant columns

# -1 means outliers, 1 means normal points
outliers = df[df['anomaly'] == -1]
print(outliers)
```

**When to Use:**
  - Your data is multidimensional, and simple techniques like Z-score or IQR might not detect outliers in complex patterns.
  - You want to handle multivariate outliers that can't be identified by looking at individual columns.

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**
DBSCAN is another unsupervised algorithm that groups data into clusters. Points that don’t fit well into any cluster are treated as outliers.


### 7. Leave Outliers Unchanged

Sometimes, outliers may contain important information (e.g., in fraud detection or rare event analysis). In such cases, it’s important to leave them as they are.

**When to Use:**
  - When the outliers are valid observations and represent rare but important events.
  - You’re working on a problem where extreme values are expected (e.g., in finance, fraud detection).


### Best Practices for Handling Outliers
  1. Understand the Context: Always investigate the cause of outliers before deciding to remove or modify them. Outliers could be mistakes, or they could be valuable information.

  2. Visualize the Data: Use visual methods (box plots, scatter plots, histograms) to detect and understand the nature of outliers.

  3. Don’t Always Remove Outliers: Removing outliers may simplify your analysis, but you could lose important insights, especially in fields where rare events matter.

  4. Use Domain Knowledge: Understanding the domain (business, medical, financial, etc.) will help you decide if outliers are anomalies or important observations.


### Conclusion:
Handling outliers depends on the nature of the data and the problem you're solving. Techniques like removal, imputation, capping, and transformation help control the influence of outliers, while machine learning methods like Isolation Forest or DBSCAN provide more advanced tools for detecting outliers in multidimensional datasets.
