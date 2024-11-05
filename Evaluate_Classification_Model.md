# Evaluate Classification Model

In machine learning, accuracy, precision, recall, and F1 score are metrics commonly used to evaluate classification models. Here’s a breakdown of each:


Imagine you built a model to predict if an email is spam or not spam. Here are some basic definitions, followed by how each metric works with an example.


### Confusion Matrix:
A confusion matrix is a table that helps to visualize the performance of a classification model by showing the actual vs. predicted classifications. It breaks down how many correct and incorrect predictions the model made across each class.

Let's go through the structure of a confusion matrix with an example.

#### Structure of a Confusion Matrix
For a binary classification problem (e.g., predicting if an email is spam or not spam), the confusion matrix is a 2x2 table, structured like this:

|  | Predicted: Positive (Spam) | Predicted: Negative (Not Spam) |
|--| -------------------------- | ------------------------------ |
| Actual: Positive (Spam) | True Positive (TP) | False Negative (FN) |
| Actual: Negative (Not Spam) | False Positive (FP) | True Negative (TN) |


#### Definitions:
- **True Positive (TP):** The model correctly predicts an email as spam when it is indeed spam.
- **True Negative (TN):** The model correctly predicts an email as not spam when it is indeed not spam.
- **False Positive (FP) (Type I Error):** The model incorrectly predicts an email as spam when it is actually not spam (also known as a "false alarm").
- **False Negative (FN) (Type II Error):** The model incorrectly predicts an email as not spam when it is actually spam (missed detection).



#### Example Confusion Matrix:
Suppose you tested your model on 100 emails, and the outcomes are as follows:

|    | Predicted: Spam | Predicted: Not Spam |
|----|-----------------|---------------------|
| Actual: Spam | 30 (TP) | 10 (FN) |
| Actual: Not Spam | 5 (FP) | 55 (TN) |

In this example:

- **True Positives (TP)** = 30: The model correctly classified 30 spam emails as spam.
- **True Negatives (TN)** = 55: The model correctly classified 55 non-spam emails as not spam.
- **False Positives (FP)** = 5: The model incorrectly flagged 5 non-spam emails as spam.
- **False Negatives (FN)** = 10: The model missed 10 spam emails, marking them as not spam.


#### Metrics Calculation

1. **Accuracy:**
  - **Definition:** The percentage of correct predictions out of all predictions.
  - **Formula:** Accuracy = TP+TN/TP+TN+FP+FN
  - **Calculation:** 30+55/30+55+5+10 = 30+55/100 = 0.85 OR 85%
  - **Interpretation:** 85% of all emails were classified correctly, whether as spam or not spam.

2. **Precision (for spam class):**
  - **Definition: The percentage of correctly predicted spam emails out of all emails predicted as spam.
  - Formula: Precision = TP/TP+FP
  - Calculation: 30/30+5 = 0.857 or 85.7%
  - Interpretation: When the model says an email is spam, it’s correct 85.7% of the time. Precision is especially important in cases where false positives are costly, like in spam detection where users don’t want non-spam emails flagged.

3. **Recall (for spam class):**
  - Definition: The percentage of actual spam emails that were correctly identified by the model.
  - Formula: Recall = TP/TP+FN
  - Calculation: 30/30+10 = 0.75 or 75%
  - Interpretation: The model correctly identified 75% of the actual spam emails. Recall is important when missing a positive case (like a spam email) has consequences, for instance, missing a critical alert.

4. **F1 Score:**
  - Definition: The harmonic mean of precision and recall, balancing the two metrics. It’s useful when you need a single metric that accounts for both false positives and false negatives.
  - Formula: F1 Score = 2 x (Precision x Recall)/(Precision + Recall)
  - Calculation: 2 x (0.857 x 0.75)/(0.857 + 0.75) = ~ 0.799 or 79.9% (approximate value)
  - Interpretation: The F1 score of 79.9% suggests a balance between precision and recall. This is useful if you want a fair trade-off between catching most spam emails (recall) and minimizing false alarms (precision).


#### Summary
  - Accuracy gives a general idea of correctness but can be misleading if classes are imbalanced (e.g., many more "not spam" than "spam" emails).
  - Precision focuses on how many spam predictions were actually correct, valuable when false positives are an issue.
  - Recall focuses on how many real spam emails were caught, valuable when false negatives (missed spam) are a concern.
  - F1 Score provides a balanced measure, useful when you need to balance precision and recall.


In this example, the model performs reasonably well with high accuracy and precision, but slightly lower recall, meaning it misses some spam emails. The F1 score of 79.9% reflects this balance.
