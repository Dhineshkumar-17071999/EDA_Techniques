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

