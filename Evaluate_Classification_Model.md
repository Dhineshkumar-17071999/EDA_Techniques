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
