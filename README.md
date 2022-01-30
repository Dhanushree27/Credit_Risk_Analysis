# Credit Risk Analysis

## Overview

This analysis was undertaken with the intention to apply machine learning to perform credit card risk analysis and predict high risk and low risk applications, based on a credit card credit dataset obtained from LendingClub, a peer-to-peer lending services company.

Since credit risk is an imbalanced dataset by nature, dfferent oversampling and undersampling techniques were applied to the dataset to resolve class imbalance, followed by application of a Logistic Regression model to predict the Risk. For oversampling, Random Oversampling and Synthetic Minority Oversampling Technique (SMOTE) were used. For undersampling, Cluster Centroid Undersampling technique was applied. In addition to these, the combination technique SMOTEENN which uses Smote oversampling, followed by undersampling to reduce influence of outliers was also applied.

Apart from resampling of data, ensemble models that reduce bias were also applied on the dataset. From the imblearn library, a random forest model, BalancedRandomForestClassifier was used and a boosting model (AdaBoost), EasyEnsembleClassifier was used. The performance of the different models have been discussed below.  

For the analysis, the initial dataset was imported as a dataframe and cleaned. The text values in the dataframe were converted to numerical values using get dummies method in pandas.Out of the 86 columns, the conversion was applied to 9 columns resulting in a final dataset of 96 columns and 68817 rows. All columns except _loan status_ were considered as features and the target was _loan status_. The data was then split into training and testing using the train_test_split method

## Results

### Resampling techniques

#### Oversampling

**RandomOverSampler:**
RandomOverSampler was imported from imblearn library. The model was used to resample the data and then logistic regression using solver 'lbfgs' was applied to predict the outcome. Running the metrics on the resultant data revealed a 65% accuracy, 1% precision for high risk and 100% for low risk, and 74% sensitivity for high risk. The f1 score was very low at 2% for high risk. From the metrics, we can see that the model was **not successful** at predicting the outcome. Oversampling using this method did not prove to be effective since the precision is very low resulting in a lot of false positives.

![RandomOverSampler](https://github.com/Dhanushree27/Credit_Risk_Analysis/blob/main/images/RandomOverSampler.PNG)

**SMOTE:**
Similar to RandomOverSampler, another over sampling technique SMOTE was applied and logistic regression was used to predict the outcome. The sampling strategy was set as 'auto'. The metrics revealed 66% accuracy, 1% precision for high risk and 100% for low risk, and 63% sensitivity for high risk. The f1 score was at 2% for high risk. Though the accuracy is slightly higher, the precision is equally bad with a lower sensitivity/ recall. Therefore, the model was **not successful** at predicting the outcome.

![SMOTE](https://github.com/Dhanushree27/Credit_Risk_Analysis/blob/main/images/SMOTE.PNG)

#### Undersampling

**ClusterCentroids:**
For undersampling, the ClusterCentroids method was applied. This did not result in higher metrics and performed worse than Oversampling at accuracy. The metrics are 54% accuracy,  1% precision for high risk and 100% for low risk, and 69% sensitivity for high risk. The f1 score was even lower at 1% for high risk. Therefore, we can conclude that the model was **not successful** at predicting the outcome.

![ClusterCentroids](https://github.com/Dhanushree27/Credit_Risk_Analysis/blob/main/images/ClusterCentroids.PNG)

#### Combination Sampling

**SMOTEENN:**
The combination sampling technique SMOTEENN was used. Despite the use of a combination technique to reduce influence of any outliers, there was no considerable improvement. The metrics are 64% accuracy,  1% precision for high risk and 100% for low risk, and 68% sensitivity for high risk. The f1 score was low at 2% for high risk. Again, the model was **not successful** at predicting the outcome.

![SMOTEENN](https://github.com/Dhanushree27/Credit_Risk_Analysis/blob/main/images/SMOTEENN.PNG)

### Ensemble Techniques
Another approach was to use models that reduce bias on the dataset. The models were imported from the imblearn library

**BalancedRandomForestClassifier:**
With the use of a random forest technique, the results were slightly better with 79% accuracy,  3% precision for high risk and 100% for low risk, and 0.70 sensitivity for high risk. The f1 score slightly higher at 6% for high risk. Despite the increase in accuracy, precision, and f1 score by a small amount, the results were not considerable enough to make an impact since the precision is still very low. Therefore, this ensemble technique was **not successful** at predicting the outcome.The analysis of feature importance, revealed that the top feature is 'total_rec_prncp' followed by 'total_pymnt'. A lot of the features make a minor contribution with about 11 features not contributing. 

![RandomForest](https://github.com/Dhanushree27/Credit_Risk_Analysis/blob/main/images/RandomForest.PNG)

**EasyEnsembleClassifier:**
EasyEnsembleClassifier is an adaptive boosting technique and provided better results. The accuracy was 93% followed by 9% precision for high risk and 100% for low risk, and 92% sensitivity. It resulted in a higher f1 score of 16%. Despite the increase in performance, the model still cannot be considered successful since there are a still a large number of false positives, but the sensitivity is much higher suggesting that the model is successul at identifying most of the high risk loans. 

![EasyEnsemble](https://github.com/Dhanushree27/Credit_Risk_Analysis/blob/main/images/EasyEnsemble.PNG)

## Summary

Different resampling and ensemble techniques were applied, but none were completely successful. The resampling techniques performed similar to each other, with around 65% accuracy and very low precision rate and did not result in any successful model. The undersampling method performed the poorest amongst all models. The better performing model was the ensemble technique, _Easy Ensemble Classifier_ that resulted in a accuracy of 93% and sensitivity of 92%, but the precision was still very low at 9%. Therefore, the Easy Ensemble Classifier cannot be considered completely successful.
Though Easy Ensemble Classifier is performing well, it cannot be recommended. In terms of sensitivity it captures 92% of high risk loans, which is a favorable aspect of the model, since sensitivity is more valuable than precision for this particular problem; but, the precision rate, and f1 score are quite low at 9% and 16% respectively. This means that there will be a large number of false positives for high risk, resulting in a lot of loans getting classified as high risk. A model with a more balanced, or atleast a slighly better performance on precision would be more valuable.









