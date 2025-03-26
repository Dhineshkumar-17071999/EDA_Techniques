# EDA_Techniques

## 1. Understanding the Dataset
- Load the dataset (pandas.read_csv(), pandas.read_excel(), etc.)
- Display the first few rows (df.head())
- Check dataset structure (df.info())
- Identify numerical & categorical features (df.dtypes)
- Check for missing values (df.isnull().sum())
- Identify duplicate rows (df.duplicated().sum())


## 2. Data Cleaning
- Handle missing values:
    - Drop missing values (df.dropna())
    - Impute missing values (mean, median, mode, forward fill, backward fill)
- Remove duplicate rows (df.drop_duplicates())
- Fix incorrect or inconsistent data entries
- Standardize column names for consistency

## 3. Univariate Analysis (Single Variable Analysis)
**For Numerical Features:**
- Summary statistics (df.describe())
- Visualizations:
    - Histogram (sns.histplot())
    - Box plot (sns.boxplot())
    - Kernel Density Estimate (KDE) plot (sns.kdeplot())
- Detect outliers using:
    - Box plot method (IQR)
    - Z-score method
 
**For Categorical Features:**
- Count of each category (df['column'].value_counts())
- Visualizations:
    - Bar plot (sns.countplot())

## 4. Bivariate Analysis (Relationship Between Two Variables)
**Numerical vs. Numerical:**
- Scatter plot (sns.scatterplot())
- Correlation matrix (df.corr())
- Heatmap (sns.heatmap())

**Numerical vs. Categorical:**
- Box plot (sns.boxplot(x='category_col', y='num_col'))
- Violin plot (sns.violinplot())

**Categorical vs. Categorical:**
- Cross-tabulation (pd.crosstab())
- Stacked bar chart


## 5. Multivariate Analysis (More Than Two Variables)
- Pair plot (sns.pairplot())
- Heatmap of correlation (sns.heatmap(df.corr()))
- Principal Component Analysis (PCA)
- Clustering (e.g., K-Means)

## 6. Outlier Detection & Handling
- Detect Outliers Using:
    - Box plot method (IQR)
    - Z-score method (scipy.stats.zscore())
    - Histogram distribution
- Handling Outliers:
    - Remove outliers
    - Cap or floor extreme values
    - Apply log transformation


## 7. Feature Engineering
- Create new features (e.g., ratios, date-based features)
- Encode categorical variables:
    - One-Hot Encoding (pd.get_dummies())
    - Label Encoding (sklearn.preprocessing.LabelEncoder)
- Binning numerical data
- Handle skewed data (log transformation, Box-Cox transformation)

## 8. Feature Selection
**Filter Methods:**
- Correlation matrix
- Chi-Square test
- Variance threshold

**Wrapper Methods:**
- Recursive Feature Elimination (RFE)
- Forward Selection
- Backward Elimination

**Embedded Methods:**
- Lasso Regression (L1 Regularization)
- Feature importance from Random Forest / XGBoost

**Dimensionality Reduction Techniques:**
- Principal Component Analysis (PCA)
- t-SNE, UMAP

## 9. Feature Scaling & Transformation
- Standardization (StandardScaler())
- Normalization (MinMaxScaler())
- Log transformation (for skewed data)


## 10. Data Visualization & Final Insights
- Summarize key findings
- Identify patterns & anomalies
- Prepare data for modeling



