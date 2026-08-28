# Bank customer churn prediction project 
Machine learning project focused on predicting bank customer churn using XGBoost and translating the model results into business insights and retention strategies.

## 1.- Business problem 
The goal of this project is to predict which customers are likely to leave the bank and identify the main factors associated with customer churn. This could help the bank identify high-risk customers and take targeted retention actions to reduce customer loss.

## 2.- Dataset 
### 2.1.- Dataset understanding 
The dataset is publicly available on Kaggle and contains information about bank customers, including demographic, financial, and banking activity information, as well as whether the customer left the bank.

*Dataset source:* [Dataset in Kagge](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)

The dataset contains 10,000 customers and 18 variables initially.

| Variable | Description |
|---|---|
| `RowNumber` | Record number. Each row represents a customer. |
| `CustomerId` | Unique customer identifier. |
| `Surname` | Customer's surname. |
| `CreditScore` | Customer's credit score. |
| `Geography` | Customer's country. |
| `Gender` | Customer's gender. |
| `Age` | Customer's age. |
| `Tenure` | Number of years the customer has been with the bank. |
| `Balance` | Customer's account balance. |
| `NumOfProducts` | Number of bank products used by the customer. |
| `HasCrCard` | Whether the customer has a credit card. |
| `IsActiveMember` | Whether the customer is an active bank member. |
| `EstimatedSalary` | Customer's estimated salary. |
| `Exited` | Whether the customer left the bank. **Target variable.** |
| `Complain` | Whether the customer has made a complaint. |
| `Satisfaction Score` | Customer satisfaction score regarding complaint resolution. |
| `Card Type` | Type of credit card held by the customer. |
| `Point Earned` | Points earned through credit card usage. |

### 2.2.- Data cleaning 
The structure of the dataset was first examined. The dataset contains 10,000 entries representing customers and 18 columns initially.

Three columns were removed because they do not provide useful information for predicting customer churn:

- RowNumber
- CustomerId
- Surname

The dataset was also checked for duplicate rows and missing values. No duplicates or missing values were found.

## 3.- Exploratory Data Analysis 
### 3.1 Target Variable

The target variable, Exited, shows a clear class imbalance: Approximately 80% of customers stayed with the bank. Approximately 20% of customers left. 

This imbalance was taken into consideration during model training and evaluation.

### 3.2 Numerical Variables

The numerical variables were examined using summary statistics. No concerning outliers or clearly invalid values were identified.

Then, mean values were then compared between customers who stayed and those who left. The main differences were observed in:

Age: Customers who left were, on average, approximately 7.4 years older than customers who stayed.
Balance: Customers who left had, on average, approximately 18,400 higher account balances.
FOTO!

These variables were therefore identified as particularly interesting for further analysis.

### 3.3 Categorical Variables

Categorical variables were analyzed by comparing churn rates across their different categories.

Noticeable differences were observed for variables such as:
FOTO!

- Geography
- Gender
- IsActiveMember

The strongest relationship was found between Complain and Exited. Approximately 99.5% of customers who made a complaint left the bank. This was also reflected in the correlation matrix, where Complain and Exited showed an extremely strong correlation of approximately 1.

This unusually strong relationship was investigated further because it could indicate potential data leakage or a specific characteristic of how the dataset was constructed.

## 4.- Feature engineering 
The target variable, Exited, was separated from the predictor variables.

One-hot encoding was applied to the categorical variables:

- Geography
- Gender
- Card Type

This converted the categorical variables into numerical features suitable for the XGBoost model.

A new binary feature, HasBalance, was also created to indicate whether a customer had a positive account balance. This was motivated by the observation during the EDA that a large number of customers had a balance of zero.

## 5.- Train/Test split 
The dataset was split into:

- 80% training data
- 20% testing data

Stratification was applied to maintain a similar proportion of customers who stayed and left in both datasets.

Because the target variable is imbalanced, scale_pos_weight was used during model training to give more importance to the minority class (Exited = 1).

## 6.- Model selection and training 
XGBoost was selected because it performs well on structured/tabular data and can capture non-linear relationships between variables. It also does not require feature scaling and provides useful feature importance information, making it suitable for both prediction and interpretation.

The main model parameters were:

- n_estimators = 200
- learning_rate = 0.05
- max_depth = 4
- scale_pos_weight = calculated from the training data
- random_state = 42

## 7.- Model evaluation 
### 7.1.- Initial model using Complain
An initial XGBoost model was trained including the Complain variable.

The model achieved almost perfect performance. Further analysis showed that Complain accounted for approximately 98% of the model's feature importance and was almost perfectly associated with the target variable.

This relationship appeared unusually strong and could indicate potential data leakage or an overly direct relationship between the feature and the target.

However, Complain can also be interpreted as a potentially important real-world signal, since customers who make complaints may have a higher probability of leaving.

To avoid relying on a potentially problematic feature, Complain was excluded from the final model. This allowed the model to evaluate whether customer churn could be predicted using the other customer characteristics.

### 7.2.- Final model results 
The final XGBoost model, without Complain, achieved: 

| Metric | Score |
|---|---|
| `Accuracy` | 80.9% |
| `Precision` | 52.1% |
| `Recall` | 77.9% |
| `F1-Score` | 62.5% |
| `ROC-AUC` | 87.8% |

### 7.3.- Metric's results interpretation 

*Accuracy – 80.9%*

The model correctly classified approximately 81% of customers as either staying or leaving. However, accuracy is not the main metric in this case because the target variable is imbalanced, with approximately 80% of customers staying and 20% leaving.

*Precision – 52.1%*

Of the customers predicted as likely to churn, approximately 52% actually left the bank.

The relatively low precision means that some customers identified as high-risk would not actually churn. However, this does not necessarily make the model ineffective. Its usefulness depends on the cost and effectiveness of the retention strategy. If contacting at-risk customers is relatively inexpensive, targeting a broader group may still be valuable.

*Recall – 77.9%*

The model correctly identified approximately 78% of customers who actually churned.

This is particularly relevant from a business perspective because failing to identify a customer who is going to leave represents a missed opportunity for retention.

*F1-score – 62.5%*

The F1-score provides a balance between precision and recall. It reflects the trade-off between identifying as many churners as possible while avoiding too many false positives.

*ROC-AUC – 87.8%*

The model shows good ability to distinguish between customers who churn and customers who stay. A value closer to 1 indicates better discrimination, while 0.5 represents performance similar to random classification.

### 7.4.- Business interpretation 
The model prioritizes identifying customers at risk of churn, achieving a relatively high recall of 77.9%.

Although precision is lower at 52.1%, this does not necessarily make the model unsuitable for business use. The appropriate balance between precision and recall depends on the cost of retention actions.

For example, if a retention campaign is relatively inexpensive, the bank may prefer to contact a larger number of potentially at-risk customers, accepting that some of them would have stayed anyway.

## 8.- Model interpretation 
The feature importance analysis of the final model showed that the most influential predictors were:

| Feature | Importance |
|---|---|
| `NumOfProducts` | 21.6% |
| `Age` | 18.0% |
| `IsActiveMember` | 15.3% |
| `Geography_Germany` | 11.7% |
| `Gender_Male` | 7.1% |
| `Balance` | 6.1% |

NumOfProducts, Age, and IsActiveMember were particularly influential in the model's predictions. Geography, especially Germany, also contributed substantially.

These results indicate that these variables are useful for identifying patterns associated with customer churn. However, feature importance represents predictive relevance, not causation. Therefore, these variables should be interpreted as indicators of churn risk rather than direct causes of customer churn.

The HasBalance feature had zero feature importance in the final model, suggesting that the binary distinction between customers with and without a balance did not provide additional predictive information beyond the original Balance variable and the other features.

## 9.- SHAP (Explainable AI) 
SHAP (SHapley Additive exPlanations) was used to better understand how the main features influence the model's predictions. While feature importance shows which variables are most important overall, SHAP also shows the direction of their impact: values on the right increase the predicted probability of churn, while values on the left decrease it. The color indicates the feature value, with red representing higher values and blue representing lower values.

The SHAP analysis provides several relevant insights:

Age: Higher age values (red) tend to have a positive impact on the predicted probability of churn, while lower values (blue) tend to reduce it. This indicates that older customers are generally predicted to have a higher churn risk.

NumOfProducts: This variable shows a non-linear relationship with churn. Lower and moderate values can have a positive impact on the prediction, while some higher values substantially reduce the predicted churn risk. However, a smaller group of high values has a strong positive impact, suggesting that the relationship is not simply linear.

IsActiveMember: Active members (1, red) tend to reduce the predicted probability of churn, while inactive members (0, blue) tend to increase it. This is consistent with the EDA, where inactive customers showed a higher churn rate.
Gender_Male: Male customers (1, red) tend to have a slightly negative impact on the churn prediction, while female customers (0, blue) tend to have a slightly more positive impact.

Balance: Higher account balances (red) generally increase the predicted probability of churn, while lower balances (blue) tend to have a lower or negative impact.

Geography_Germany: Customers from Germany (1, red) tend to have a positive impact on the predicted probability of churn, while customers from other countries (0, blue) tend to have a lower impact.

Overall, the SHAP analysis provides additional interpretability to the XGBoost model by showing not only which variables are important, but also how different feature values influence the model's churn predictions. These results should be interpreted as predictive relationships rather than causal effects.

## 10.- Business recommendations 

The bank could use the model's predicted churn probability to identify customers who are more likely to leave and prioritize them for retention actions.

In addition, NumOfProducts was the most influential predictor. The bank could investigate how churn rates vary across different numbers of products and determine whether certain customer groups are more at risk. This could help identify opportunities to improve product engagement and customer value.

Also, IsActiveMember was one of the most important predictors. Customers showing low engagement could be targeted with personalized communication, relevant product recommendations, or digital engagement campaigns.

Age and geography, particularly Germany, were important predictors. The bank could analyze these segments further to understand whether differences in customer needs, products, or customer experience may be associated with higher churn.

Finally, rather than applying the same retention strategy to every customer, the bank could combine predicted churn probability with customer value. For example, high-value customers with a high predicted probability of churn could receive more personalized retention actions.

## 11.- Limitations 
This project uses a publicly available dataset and therefore does not represent the full complexity of a real banking environment.

In a real-world setting, additional information such as transaction behaviour, product usage, customer interactions, and historical changes in account activity could improve the model.
