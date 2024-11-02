# Methods to handle categorical columns:

Here are some common encoding methods for categorical columns:

1. One-Hot Encoding
  - Converts each unique category value into a new binary (0 or 1) column.
  - If you have a "Color" feature with values "Red," "Blue," and "Green," it will create three columns: Color_Red, Color_Blue, and Color_Green.
  - Useful when categories are not ordered and few in number.
  - Example: Red becomes [1, 0, 0], Blue becomes [0, 1, 0].

2. Label Encoding
  - Assigns each category a unique numerical value (e.g., "Red" = 0, "Blue" = 1, "Green" = 2).
  - Simple and memory-efficient but may create a false sense of order.
  - Useful when categories have a natural ordering (e.g., "Low," "Medium," "High").
    
3. Ordinal Encoding
  - Similar to label encoding but used when categories have a clear, meaningful order.
  - For example, in "Education Level" with categories "High School," "Bachelor’s," "Master’s," "PhD," you might encode them as 0, 1, 2, 3 respectively.
    
4. Binary Encoding
  - Combines label and binary encoding by converting categories into binary digits and using fewer columns.
  - For example, "Red" = 001, "Blue" = 010, "Green" = 011.
  - Useful when there are many categories and you want to reduce dimensionality.

5. Frequency Encoding
  - Replaces each category with the frequency of its occurrence.
  - For example, if "Red" appears 30 times, "Blue" 20 times, and "Green" 50 times, it would assign values 30, 20, and 50 respectively.
  - Useful to give weight to more common categories in the data.
    
6. Target Encoding
  - Replaces categories with the mean of the target variable for each category.
  - For example, if "Red" has an average target value of 0.8, "Blue" 0.3, and "Green" 0.6, it assigns these values to each category.
  - Often used in target-variable prediction tasks, but care is needed to prevent overfitting.
    
7. Hash Encoding
  - Uses a hash function to convert categories into numerical values, which helps reduce dimensionality without creating a column for each unique category.
  - Useful for datasets with many categories but may create hash collisions (where two categories map to the same value).
    
8. Mean Encoding
  - Similar to target encoding, but instead of using the target mean, you calculate the mean of another feature for each category.
  - For example, you might encode "Red," "Blue," and "Green" based on the mean age of people in each color category.
    
9. Count Encoding
  - Replaces each category with the count (number of occurrences) of that category in the dataset.
  - For example, if "Red" appears 100 times, it will be replaced with 100.
  - Useful when the frequency of a category is meaningful.

