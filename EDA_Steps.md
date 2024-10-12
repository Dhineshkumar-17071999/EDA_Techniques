# EDA_Techniques

### Key components of EDA include:

1. Introduction to EDA:
    - Purpose and goals of EDA.
    - Understanding different data types (Numerical, Categorical, etc.).
    - Tools and libraries used for EDA.

2. Data Collection and Importing:
    Before starting with EDA, you first need to acquire the data from various sources like CSV, Excel, SQL databases, or APIs.
    - Data Sources: The data can come from multiple sources such as:
        - Flat files (e.g., CSV, JSON, Excel)
        - Databases (SQL, NoSQL)
        - Databases (SQL, NoSQL)
        - Pre-existing datasets (Kaggle, UCI repository)
    
    - Tools for Importing: Common tools and libraries used for importing data include:
        - Python: pandas, sqlite3, requests
        - Loading data into a DataFrame using pandas.
        - Inspecting the data: .head(), .info(), .describe(), etc.
        - Checking data types and basic structure.

3. Data Cleaning:
    Once the data is imported, it is important to clean it before analysis. Data cleaning refers to the process of handling issues like missing values, incorrect data types, and duplicates.

    3.1. Handling Missing Data
         1. Types of Missingness: Missing data can occur in different forms:
            - MCAR: Missing Completely at Random
            - MAR: Missing at Random
            - MNAR: Missing Not at Random
         2. Strategies for Handling Missing Data:
            - Removing Missing Data: If the proportion of missing data is low, you might drop rows or columns.
            - Imputation:
                - Mean/Median Imputation: For numerical data.
                - Mode Imputation: For categorical data.
                - K-Nearest Neighbors (KNN) Imputation: A more advanced technique.
                - Forward/Backward Fill: For time series data.
    
    3.2. Handling Duplicates
        - Duplicate Rows: Sometimes datasets contain duplicate entries, which need to be removed using:
            - Python: pandas.drop_duplicates()

    3.3. Handling Outliers
        - Outliers can skew the results of data analysis. Common techniques to detect and handle outliers include:
            - Z-Score: Identifies outliers based on standard deviation.
            - IQR (Interquartile Range): Outliers are typically defined as values 1.5 times above or below the IQR.
            - Visualization Methods: Box plots and scatter plots are often used to detect outliers visually.

5. Data Transformations
    After cleaning the data, the next step is to transform it into a form that is easier to analyze.
    4.1. Normalization/Standardization
        - Normalization: Scaling data between 0 and 1. Used for features that require this, such as those used in neural networks.
        - Formula: (x-min(x))/max(x) - min(x)
        - Standardization: Scaling data so that it has a mean of 0 and standard deviation of 1. Used in algorithms like SVM and KNN.
        - Formula: x-u/sigma

    4.2. Encoding Categorical Data
        Many machine learning algorithms cannot work with categorical data directly. Methods for encoding include:
        - Label Encoding: Converts categories to numeric labels (e.g., Male = 0, Female = 1).
        - One-Hot Encoding: Converts categorical variables into multiple binary columns.One-Hot Encoding: Converts categorical variables into multiple binary columns.
            - Python: pandas.get_dummies()

    4.3. Binning
        - Discretization: For continuous variables, you can bin the data into ranges or bins (e.g., age into groups: 0-18, 19-35, etc.).
        - Equal-width Binning: Bins are of equal width.
        - Equal-frequency Binning: Each bin has the same number of data points.

    - Scaling and Normalization.
    - Logarithmic transformation to handle skewed data.

6. Descriptive Statistics:
    - Measures of central tendency: mean, median, mode.
    - Measures of dispersion: range, variance, standard deviation.
    - Skewness and kurtosis.

7. Univariate Analysis (Analyzing one variable at a time)
    Univariate analysis focuses on analyzing one variable at a time. It helps understand the distribution, central tendency, and spread of a feature.
    6.1. For Numerical Data
        - Summary Statistics:
            - Mean: Average value.
            - Median: The middle value.
            - Mode: Most frequent value.
            - Variance and Standard Deviation: Measures of dispersion.
            - Range: Difference between maximum and minimum.
            - Quantiles: Median (50th percentile), Quartiles (25th and 75th percentiles).
        - Visualizations:
            - Histogram: Shows the distribution of values.
            - Box Plot: Useful for identifying outliers and understanding the spread.
            - Box Plot: Useful for identifying outliers and understanding the spread.
        
    6.2. For Categorical Data
        - Frequency Tables: Counts of each category.
        - Bar Plot: Visual representation of frequencies.
        - Pie Chart: Proportion of each category, though less recommended for large categories.

6. Bivariate Analysis (Analyzing two variables at a time)
    - Numerical vs. Numerical: Scatter plots, Correlation heatmaps.
    - Numerical vs. Categorical: Box plots, Violin plots.
    - Categorical vs. Categorical: Cross-tabulation, Stacked bar charts.

7. Multivariate Analysis
    Multivariate analysis involves the exploration of relationships between multiple variables simultaneously.
    7.1. Numerical vs. Numerical
        - Correlation: Pearson correlation coefficient measures the linear relationship between two continuous variables. Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation).
        - Heatmap: A visual representation of correlation matrices.
        - Scatter Plot: Plots one variable against another to see if there is a relationship.
        - Pair Plot: Visualizes the pairwise relationships across multiple variables.
    
    7.2. Categorical vs. Categorical
        - Contingency Table: Summarizes the relationship between two categorical variables.
        - Chi-Square Test: Tests the independence of two categorical variables.
        - Stacked Bar Plot: Used to visualize the relationship between two categorical variables.

    7.3. Categorical vs. Numerical
        - Box Plot: Shows how numerical values differ across categories.
        - Violin Plot: Combines box plot with a density plot.
        - ANOVA (Analysis of Variance): Tests if the means of different groups are significantly different.

8. Feature Engineering
    Feature engineering is the process of creating new features or transforming existing features to better represent the underlying structure of the data.
    8.1. Creating Interaction Terms
        - Polynomial Features: Interaction terms can capture non-linear relationships between variables. These are often useful in linear models.
            - Python: PolynomialFeatures() in sklearn

    8.2. Date-Time Features
        - Extracting Components: From a date-time column, you can extract useful features such as:
            - Year, Month, Day, Hour, Day of Week, etc.
        - Time Series Decomposition: Breaking down time series data into trend, seasonality, and noise.

    - Handling categorical variables using encoding techniques (One-Hot Encoding, Label Encoding).

9. Dimensionality Reduction
    When you have a large number of features, dimensionality reduction techniques help reduce the dataset’s complexity while retaining its essential structure.
    9.1. Principal Component Analysis (PCA)
        - PCA is a technique used to reduce the dimensionality of numerical data by transforming the data into a set of orthogonal (uncorrelated) features known as principal components.

    9.2. t-SNE and UMAP
        - t-SNE (t-Distributed Stochastic Neighbor Embedding): A technique for visualizing high-dimensional data by reducing it to two or three dimensions.
        - UMAP (Uniform Manifold Approximation and Projection): Similar to t-SNE, but often faster and better at preserving global structure.

10. Visualizing Relationships
    Visualization is an integral part of EDA. Some common types of plots used in EDA include:
    - Histograms: Show the frequency distribution of a single variable.
    - Box Plots: Highlight the distribution and outliers.
    - Bar Plots: Compare the frequencies of categorical variables.
    - Scatter Plots: Reveal relationships between two continuous variables.
    - Correlation Matrix Heatmap: Visualizes correlations between numerical variables.
    - Pair Plots: Displays pairwise relationships in a dataset.
    - Using libraries like matplotlib, seaborn, and plotly for effective visualization.
    - Adding interactivity to visualizations (optional, using Plotly).

11. Hypothesis Testing
    EDA also involves validating hypotheses about your data. Some common tests include:
    - T-tests: Compare the means of two groups.
    - ANOVA (Analysis of Variance): Compare the means of more than two groups.
    - Chi-Square Test: Test for independence between categorical variables.
    - Shapiro-Wilk Test: Test for normality in a dataset.

12. Identifying Patterns, Trends, and Relationships
    EDA helps you uncover:
    - Patterns: Recurring behaviors in your data.
    - Trends: Long-term increase or decrease in the data.
    - Seasonality: Patterns that repeat over a regular time period.
    - Anomalies: Data points that do not fit the usual pattern, often outliers.

13. Data Summarization
    Finally, summarizing the dataset helps in understanding the key findings from EDA:
    - Summary Reports: A concise description of the major findings from the EDA process, such as correlations, distribution characteristics, missing data, and outliers.
    - Dashboards: Visual summaries for stakeholders, using tools like Tableau or Power BI.

12. Conclusion of EDA
    By the end of the EDA process, you should have:
    - A thorough understanding of the data’s structure, relationships, and quality.
    - Clean and transformed data that is ready for further modeling.
    - A set of hypotheses or questions that can guide the modeling phase.
