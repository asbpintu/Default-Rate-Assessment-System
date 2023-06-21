# Default-Rate-Assessment-System
This repository contains an analysis of bank loan default using machine learning techniques. The goal of this project is to build a predictive model that can accurately classify whether a borrower is likely to default on a loan based on various features.

### Business Understanding:

As an employee of a consumer finance company specializing in lending various types of loans to urban customers, Our task is to perform Exploratory Data Analysis (EDA) to identify patterns in the data. The purpose of this analysis is to ensure that the company can make informed decisions regarding loan approvals, minimizing the risk of approving loans to applicants who are unlikely to repay.

_When the company receives a loan application, it faces two types of risks in its decision-making process:_

+ Risk of Not Approving a Loan to a Creditworthy Applicant:
If an applicant is likely to repay the loan, rejecting their application would result in a loss of business for the company. Hence, it is important to identify such creditworthy applicants and approve their loans.

+ Risk of Approving a Loan to an Applicant Who is Likely to Default:
On the other hand, if an applicant is not likely to repay the loan, approving their application may lead to a financial loss for the company. It is crucial to minimize the risk of default by identifying applicants who are likely to repay.

_The following data provides information regarding the loan application at the time of submission._

## Dataset
The analysis is performed on a publicly available dataset obtained from  Bank. The dataset contains information about loan applicants, including their demographic data, financial history, and loan application details. The dataset is stored in the file `loan_applicants_details.csv`, which can be found in the [`data`](data) directory.

## Requirements
To run the analysis, you need the following dependencies:

+ Python 3.7 or higher
+ Jupyter Notebook or JupyterLab
+ Required Python libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

## Analysis Steps

**1. Data Preprocessing:** The dataset is explored and cleaned to handle missing values, outliers, and any inconsistencies. Feature engineering techniques are applied to extract relevant information from the available features.

**2. Exploratory Data Analysis (EDA):** Various visualizations and statistical analyses are performed to gain insights into the dataset. The relationships between different features and the target variable (default or non-default) are explored to identify any patterns or trends.

**3. Feature Selection:** Statistical tests, correlation analysis, and domain knowledge are used to select the most relevant features for training the predictive model. Feature importance is determined using techniques such as feature ranking or feature importance scores.

**4. Model Development:** Several machine learning models are trained using the preprocessed dataset. Different algorithms such as logistic regression, decision   trees, random forests, and gradient boosting are implemented to compare their performance. Cross-validation techniques are employed to evaluate the models and    select the best performing one.

**5. Model Evaluation:** The selected model is evaluated using various evaluation metrics such as accuracy, precision, recall, and F1-score. Additionally, a       confusion matrix is generated to analyze the performance in terms of true positives, false positives, true negatives, and false negatives.

**6. Model Deployment:** Once the best performing model is identified, it is saved and deployed for future use. This can include integrating the model into a web  application or using it for real-time predictions.


## 1. Data Preprocessing:

+ **Python Libraries**

```js
# Necessary Packages

import pandas as pd

# Set Output Display
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Filter Warnings
import warnings
warnings.filterwarnings('ignore')
```

+ **Data: Reading and Understanding**

```js
# Reading Data from csv file

root = pd.read_csv('loan_applicants_details.csv')
data = root.copy()

columns_info = pd.read_csv('columns_description.csv')
```
All csv files can be found in the [`data`](data) directory

 `Understanding some attributes:-`
 
```js
data['NAME_HOUSING_TYPE'].unique()
```
```js
data['NAME_CONTRACT_TYPE'].unique()
```
```js
data['NAME_EDUCATION_TYPE'].unique()
```
```js
finacial_data = data[['AMT_INCOME_TOTAL' , 'AMT_CREDIT' , 'AMT_ANNUITY']]
finacial_data.describe().T
```
all code can be found in the [main.ipynb](main.ipynb) file.

+ **Checking null value and make dataframe will all attributes with null value (Checking the %)**

```js
null_col = data.isnull().any()
```
```js
null_col_name = data.columns[null_col]
null_col_name
```
```js
len(data[null_col_name].columns)
```
```js
null_data = data[null_col_name]
# Making data frame containing columns with null values and percentage of null values
null_per = pd.DataFrame(null_data.isnull().sum()/len(null_data)*100) . reset_index()
null_per.columns = ['column_name','null_value_percentage']
null_per
```





















