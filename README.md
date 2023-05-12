# Credit Risk Classification using Supervised Machine Learning
Demonstration of Supervised Machine Learning techniques in identifying fradulant transaction. Module imports include sklearn for preprosessing, modeling, and classification reporting.

## Credit Risk Analysis Report

### Overview

The dataset contains 77537 entries in a dataframe with 9 numerical fields. These fields include:

* loan_size
* interest_rate
* borrower_income
* debt_to_income
* num_of_accounts
* derogatory_marks
* total_debt
* loan_status (Label)

The features for the supervised learning demonstration include all the listed fields, excluding ```loan_status``` .

### Results

Segmenting the dataset into training and test divisions allows for immediate performance metrics with accuracy scores and confusion matricies. 

Logistic Regressions of the dataset results in:

*Balanced Accuracy: 0.9520479254722232*

              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.85      0.91      0.88       619

    accuracy                           0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384

The `RandomOverSampler` module is employed to provide greater representation to minority classes. This proved to be beneficial as

*Balanced Accuracy **raised to** Score: 0.9936781215845847*

*with a significant rise in recall and observed rise in true positive identification*

              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.84      0.99      0.91       619

    accuracy                           0.99     19384
   macro avg       0.92      0.99      0.95     19384
weighted avg       0.99      0.99      0.99     19384
### Discussion

Performing a Logistic Regression of the dataset proved to be a useful tool for automating risk assessment of loan applications. Oversampling minority classes imporves the model's performance in True positives and False negatives suggesting both improved recall but no significant precision improvement. Overall the accuracy of the logistic regression model is suggestive of high confidence in deployment.