# Outliers - Explanation

An outlier is a data point that significantly differs from other observations in the dataset.
Outliers can occur due to variability in the data, measurement errors, or they might indicate something 
unique or interesting about the data. They often lie far away from the central tendency (mean, median, mode) 
of the data, and can distort statistical analyses and machine learning models if not handled properly.

Outliers are important to detect because they can:
    - Skew the results of data analysis.
    - Influence model performance.
    - Reveal important anomalies (e.g., fraud detection, rare events).

## How to Detect Outliers?

There are several techniques to detect outliers, including statistical methods and visual approaches. Below are the most common ones:

### 1. Using Z-Score (Standard Score)

The Z-score measures how far a data point is from the mean, in terms of standard deviations. If the Z-score is higher (or lower) than a certain threshold (typically +3 or -3), that data point is considered an outlier.

Formula for Z-Score:

Z = X−μ/σ

​Where:
    - X = the value of the data point,
    - μ = the mean of the dataset,
    - σ = the standard deviation of the dataset.

Example in Python:

```sh
import numpy as np
import pandas as pd

# Sample dataset
df = pd.DataFrame({'Age': [25, 30, 35, 40, 120, 28, 32, 150]})

# Sample dataset
df['Z_Score'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()

# Find outliers with Z-score above 3 or below -3
outliers = df[np.abs(df['Z_Score']) > 3]
print(outliers)
```
In this example, the values 120 and 150 for "Age" would likely be considered outliers, as their Z-scores will be far from the mean.

### 2. Using the Interquartile Range (IQR)

The IQR method is a common technique for detecting outliers. It’s based on the 25th percentile (Q1) and the 75th percentile (Q3) of the data. The IQR is the range between Q1 and Q3. Outliers are typically defined as data points that fall below:
    Q1−1.5×IQR  or above  Q3+1.5×IQR

Steps to Detect Outliers Using IQR:
    1. Calculate the Q1 (25th percentile) and Q3 (75th percentile).
    2. Compute the IQR:  IQR = 𝑄3 − 𝑄1
    3. Define the lower and upper bounds:
        Lower Bound = Q1 - 1.5 * IQR, Upper Bound = Q3 + 1.5 * IQR
    4. Any data point outside these bounds is an outlier.

Example in Python:
```sh
# Sample dataset
df = pd.DataFrame({'Fare': [10, 20, 30, 40, 500, 15, 25, 1000]})

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

# Define the bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = df[(df['Fare'] < lower_bound) | (df['Fare'] > upper_bound)]
print(outliers)
```

In this case, fares like 500 and 1000 would likely be considered outliers.

### 3. Visual Methods for Detecting Outliers

Box Plot:
A box plot (or whisker plot) is one of the most common ways to visualize the spread of data and identify outliers. In a box plot, outliers appear as points outside the whiskers of the box.
    - The Box: Represents the interquartile range (IQR).
    - The Whiskers: Extend from the box to 1.5 times the IQR. Any points outside the whiskers are considered potential outliers.

Example in Python:
```sh
import matplotlib.pyplot as plt
import seaborn as sns

# Generate boxplot
sns.boxplot(x=df['Fare'])
plt.show()
```

In the box plot, points outside the "whiskers" are marked as outliers.

Scatter Plot:

A scatter plot can help detect outliers in bivariate or multivariate data. By plotting one variable against another, you can visually spot any data points that lie far from the general cluster.


### 4. Tukey's Fences

Tukey's method extends the IQR approach by introducing the concept of "inner" and "outer" fences:
    - Inner Fences: Q1 - 1.5 * IQR and Q3 + 1.5 *IQR.
    - Outer Fences: Q1 - 3 * IQR and Q3 + 3 * IQR.

Anything beyond the outer fences is considered a severe outlier, while anything between the inner and outer fences is considered a mild outlier.

### 5. Using Machine Learning Techniques

If you're working with multidimensional data, you can use machine learning techniques like **Isolation Forest** and **DBSCAN** to detect outliers.

**Isolation Forest:**
The isolation forest algorithm isolates outliers by randomly selecting a feature and then randomly selecting a split value. It builds a tree structure and calculates how "deep" a point is in the tree. The more isolated the point, the more likely it is an outlier.

Example in Python:
```sh
from sklearn.ensemble import IsolationForest

# Sample dataset
df = pd.DataFrame({'Fare': [10, 20, 30, 40, 500, 15, 25, 1000]})

# Fit isolation forest
iso = IsolationForest(contamination=0.1)
df['anomaly'] = iso.fit_predict(df[['Fare']])

# Anomalies are labeled as -1
outliers = df[df['anomaly'] == -1]
print(outliers)
```

**When to Handle Outliers?**
Not all outliers are "bad." In some cases, outliers may represent important, valid information (like rare events). You may want to handle outliers if:
    - They result from data entry errors or inaccuracies.
    - They heavily skew your analysis or model.
    - You are using models that are sensitive to outliers (e.g., linear regression).


**Common ways to handle outliers:**
    - Remove them: If they are errors or not relevant to your analysis.
    - Cap them: Set them to a maximum or minimum threshold.
    - Transform the data: Use log transformation or other scaling techniques to reduce the impact of extreme values.
