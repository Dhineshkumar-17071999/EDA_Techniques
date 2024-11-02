# Methods to find a feature is Normally Distributed or Not

The normal distribution is also known as the Gaussian distribution or bell curve.
  - **Gaussian distribution:** Another name for the normal distribution.
  - **Bell curve:** The shape of the normal distribution when graphed.
  - **Probability bell curve:** Another name for the normal distribution.  

To determine if a feature follows a normal distribution or a non-normal distribution, you can use several methods.
  - 1. Visual Methods
      - a) Histogram
        - A histogram provides a visual overview of the data distribution.
        - For a normal distribution, the histogram will show a symmetric, bell-shaped curve centered around the mean.
          ```sh
          import matplotlib.pyplot as plt

          # Example for Age feature
          plt.hist(copy_train_df['Age'], bins=20, color='skyblue', edgecolor='black')
          plt.title("Histogram of Age")
          plt.xlabel("Age")
          plt.ylabel("Frequency")
          plt.show()
          ```

      - b) Density Plot (KDE Plot)
        - A kernel density estimation (KDE) plot is a smooth version of a histogram and can show whether data has a bell-shaped curve.
          ```sh
          import seaborn as sns

          sns.kdeplot(copy_train_df['Age'], shade=True)
          plt.title("Density Plot of Age")
          plt.xlabel("Age")
          plt.show()
          ```

      - c) Q-Q Plot (Quantile-Quantile Plot)
        - A Q-Q plot compares the quantiles of your data to the quantiles of a normal distribution.
        - If the points lie approximately along a straight diagonal line, the data is likely normally distributed.
        ```sh
        import scipy.stats as stats

        stats.probplot(copy_train_df['Age'], dist="norm", plot=plt)
        plt.title("Q-Q Plot of Age")
        plt.show()
        ```
  - 2. Statistical Tests
      - a) Shapiro-Wilk Test
        - This test checks the null hypothesis that the data is normally distributed.
        - If the p-value is less than 0.05, you can reject the null hypothesis, meaning the data is not normally distributed.
        ```sh
        from scipy.stats import shapiro

        stat, p_value = shapiro(copy_train_df['Age'])
        print(f'Shapiro-Wilk Test: Statistic={stat}, p-value={p_value}')
        ```

      - b) Anderson-Darling Test
        - The Anderson-Darling test also checks for normality and is similar to Shapiro-Wilk but slightly more powerful.
        - If the test statistic exceeds the critical value, the data does not follow a normal distribution.
        ```sh
        from scipy.stats import anderson

        result = anderson(copy_train_df['Age'])
        print('Anderson-Darling Test Statistic:', result.statistic)
        for i in range(len(result.critical_values)):
            print(f'{result.significance_level[i]}%: {result.critical_values[i]}')
        ```

      - c) Kolmogorov-Smirnov Test
        - This is a more general test that compares the sample distribution with a normal distribution.
        ```sh
        from scipy.stats import kstest

        stat, p_value = kstest(copy_train_df['Age'], 'norm')
        print(f'Kolmogorov-Smirnov Test: Statistic={stat}, p-value={p_value}')
        ```

    
