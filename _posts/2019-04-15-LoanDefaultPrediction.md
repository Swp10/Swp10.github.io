---
title: "Loan Default Prediction"
date: 2019-04-15
tages: [machine learning, data science, data modeling, numpy, pandas, matplotlib, scikit-learn]
excerpt: "Machine Learning, Data Science, Data Modeling, Numpy, Pandas, Matplotlib, Scikit-learn"
gallery:
  - url: /images/loans.jpg
    image_path: /images/loans.jpg
    alt: "title image"
    title: "Loan"
---

{% include gallery id="gallery" layout="full" %}

I will be exploring publicly available Lending Club data from Kaggle. Lending Club is a platform bringing borrowers and investors together, transforming the way people access credit. As an investor, you would want to invest in people who showed a profile of having a high probability of paying back. I will try to create a model to predict this.

The features represent as follow:

1. credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
2. purpose: The purpose of the loan
3. int.rate: The interest rate of the loan
4. installment: The monthly installments owed by the borrower if the loan is funded.
5. log.annual.inc: The natural log of the self-reported annual income of the borrower.
6. dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
7. fico: The FICO credit score of the borrower.
8. days.with.cr.line: The number of days the borrower has had a credit line.
9. revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
10. revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
11. inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
12. delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
13. pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or

## Import Libraries
Import the usual libraries for pandas, plotting, and sklearn.

```python
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")
```

## Get the Data
```python
# load dataset
loan = pd.read_csv("loandata.csv")
```

## Data Exploration
Check out the info(), describe() and head() methods on loan
```python
loan.info()
loan.describe()
loan.head()
```

Check the missing data
```python
loan.isnull().sum()
```
credit.policy          0
purpose                0
int.rate              11
installment           10
log.annual.inc        15
dti                    6
fico                   8
days.with.cr.line      9
revol.bal             12
revol.util            24
inq.last.6mths         2
delinq.2yrs          141
pub.rec                7
not.fully.paid         0
dtype: int64

Check the size of the dataset
```python
loan.shape
```
(9578, 14)

Check the features of the dataset
```python
loan.columns
```

Group the not.fully.paid feature
```python
loan.groupby('not.fully.paid').size()
```
```python
#Positive examples (1) = 1533
#Negative examples (0) = 8045
# Apparently, the data set is imbalanced the examples of not.fully.paid is only 16%
```

Data correlation matrix - The correlation matrix is an important tool to understand the correlation between the different characteristics. The values range from -1 to 1 and the closer a value is to 1 the bettere correlation there is between two characteristics. Let's calculate the correlation matrix for our dataset.
```python
corr = loan.corr()
corr  
```    

Histogram is use to find the data distribution of the features, find out if there is any outliners, and observe duplicate data
```python
loan.groupby('not.fully.paid').hist(figsize=(9, 9))  
```

## Data Clean Up
```python
# select the columns where missing values and locate the zeros as a mask
mask = loan[['int.rate', 'installment', 'log.annual.inc', 'dti','fico', 'days.with.cr.line',
             'revol.bal','revol.util','inq.last.6mths','delinq.2yrs','pub.rec']] == 0
```
```python
# replace the zeros with np.nan
loan[mask] = np.nan
loan.head(5)
```
```python
#Get the mean
loan.mean()
```
```python
#Get the median
loan.median()
```

## Exploratory Data Analysis

```python
sns.boxplot(x=loan["not.fully.paid"], y=loan["int.rate"])
```
not.fully.paid loaner has a higher interest rate

```python
sns.boxplot(x=loan["not.fully.paid"], y=loan["fico"])
```
not.fully.paid loaner has a lower Fico score

```python
sns.jointplot(x='fico',y='int.rate',data=loan, color='green')
```

```python
# Calculate the median value
# Substitute it in the column of the dataset where values are NaN
median_int_rate = loan['int.rate'].median()
loan['int.rate'] = loan['int.rate'].replace(to_replace=np.nan, value=median_int_rate)
```
```python
median_installment = loan['installment'].median()
loan['installment'] = loan['installment'].replace(to_replace=np.nan, value=median_installment)
```
```python
median_LAI = loan['log.annual.inc'].median()
loan['log.annual.inc'] = loan['log.annual.inc'].replace(to_replace=np.nan, value=median_LAI)
```
```python
median_dti = loan['dti'].median()
loan['dti'] = loan['dti'].replace(to_replace=np.nan, value=median_dti)
```
```python
median_fico = loan['fico'].median()
loan['fico'] = loan['fico'].replace(to_replace=np.nan, value=median_fico)
```
```python
median_days = loan['days.with.cr.line'].median()
loan['days.with.cr.line'] = loan['days.with.cr.line'].replace(to_replace=np.nan, value=median_days)
```
```python
median_bal = loan['revol.bal'].median()
loan['revol.bal'] = loan['revol.bal'].replace(to_replace=np.nan, value=median_bal)
```
```python
median_util = loan['revol.util'].median()
loan['revol.util'] = loan['revol.util'].replace(to_replace=np.nan, value=median_util)
```
```python
median_6mths = loan['inq.last.6mths'].median()
loan['inq.last.6mths'] = loan['inq.last.6mths'].replace(to_replace=np.nan, value=median_6mths)
```
```python
median_delinq = loan['delinq.2yrs'].median()
loan['delinq.2yrs'] = loan['delinq.2yrs'].replace(to_replace=np.nan, value=median_delinq)
```
```python
median_pub = loan['pub.rec'].median()
loan['pub.rec'] = loan['pub.rec'].replace(to_replace=np.nan, value=median_pub)
```
```python
loan['not.fully.paid'].value_counts()
```

## Apply one hot encoding
```python
list(loan["purpose"].unique())
```
```python
onehot_purpose = pd.get_dummies(loan, columns=["purpose"])
onehot_purpose.head()
```

## Feature Engineering
```python
onehot_purpose.columns
```

```python
#y is the valuable we are looking for
feature_names = ['credit.policy', 'int.rate', 'installment', 'log.annual.inc', 'dti',
       'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
       'inq.last.6mths', 'delinq.2yrs', 'pub.rec',
       'purpose_all_other', 'purpose_credit_card',
       'purpose_debt_consolidation', 'purpose_educational',
       'purpose_home_improvement', 'purpose_major_purchase',
       'purpose_small_business']
X = onehot_purpose[feature_names].values
y = onehot_purpose['not.fully.paid'].values
```
## Split data into train and test sets
```python
# split dataset into test/train in 80% /20%
# random_state shuffle the same way everytime
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=35, test_size=0.2)
X_train.shape, X_test.shape
```

## Standardize the data
```python
# scale/standardize features
#why we don't call fit on test data?
# fit_transform - fit get the mean of the std for this dataset. Transform - take the mean and calculate the value  
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

X_train[:5]

X_train_std[:5]
```

## SVM Classifier
```python
svm = SVC(kernel='linear')
svm.fit(X_train_std, y_train)
pred_svm = svm.predict(X_test_std)
svm.score(X_test_std, y_test)
```
```python
#model performed
print(classification_report(y_test,pred_svm))
print(confusion_matrix(y_test, pred_svm))
```

## Logistic Classifier
```python
log_reg = LogisticRegression()
logreg = log_reg.fit(X_train_std, y_train)
pred_logreg = log_reg.predict(X_test_std)
logreg.score(X_test_std, y_test)
```
```python
#log_reg.score(X_test, y_test)
#model performed
print(classification_report(y_test,pred_logreg))
print(confusion_matrix(y_test, pred_logreg))
```

## Decision Tree Classifier
```python
tree = DecisionTreeClassifier()
Dtree= tree.fit(X_train_std, y_train)
pred_Dtree = Dtree.predict(X_test_std)
tree.score(X_test_std, y_test)
```
```python
#model performed
print(classification_report(y_test,pred_Dtree))
print(confusion_matrix(y_test, pred_Dtree))
```

## KNeighborsClassifier
```python
knn = KNeighborsClassifier(n_neighbors=7)
KNC = knn.fit(X_train_std, y_train)
pred_KNC = KNC.predict(X_test_std)
knn.score(X_test_std, y_test)
```

```python
#model performed
print(classification_report(y_test,pred_KNC))
print(confusion_matrix(y_test, pred_KNC))
```

## RandomForestClassifier
```python
rfc = RandomForestClassifier(n_estimators=200)
rfc = rfc.fit(X_train_std, y_train)
pred_rfc = rfc.predict(X_test_std)
rfc.score(X_test_std, y_test)
```

```python
#model performed
print(classification_report(y_test,pred_rfc))
print(confusion_matrix(y_test, pred_rfc))
```

## Cross Validation
```python
# 10-fold cross validation with a list of algorithms

classifiers = [svm, log_reg, Dtree, KNC, rfc]

model_scores = []
for clf in classifiers:
    model_scores.append(cross_val_score(clf, X_train_std, y_train, scoring='accuracy', cv=10))
```
```python
# use a DataFrame to view the cross validation results

models_df = pd.DataFrame(model_scores, columns=[1,2,3,4,5,6,7,8,9,10],
                               index=["SVM", "LR", "DTree", "KNC", "Forest"])
models_df
```
```python
# add a "Mean" column to the end of the DataFrame

models_df["Mean"] = models_df.mean(axis=1)
models_df
```

## Boxplot and Model Selection
```python
# BOXPLOT - visually comparing performance of the models

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))
# rectangular box plot
bplot_models = axes.boxplot(model_scores, vert=True, patch_artist=True)

# fill with colors - Models
colors_d = ["lightgreen", "lightyellow", "lime", "yellow", "yellowgreen"]
for patch, color in zip(bplot_models['boxes'], colors_d):
    patch.set_facecolor(color)

    # adding axes labels
axes.yaxis.grid(True)
axes.set_xticks([y+1 for y in range(len(model_scores))])
axes.set_xlabel('Classification Models', fontsize=18)
axes.set_ylabel('Accuracy', fontsize=18)
#axes.set_ylim((0.7, 0.85))
axes.set_title('Classification Accuracy using All Features', fontsize = 18)

# add x-tick labels
plt.setp(axes, xticks=[y+1 for y in range(len(model_scores))],xticklabels=['SVM', 'LR', 'DTree', 'KNC', 'Forest'])

# increase tick size
y_ticks = axes.get_yticklabels()
x_ticks = axes.get_xticklabels()

for x in x_ticks:
    x.set_fontsize(18)       
for y in y_ticks:
    y.set_fontsize(18)
```

## Hyperparameter Tuning (Grid Search)
```python
# hyperparameter tuning can be done manually or using Grid Search
# GridSearch returns the best model from among the various given hyperparameters

# Grid Search
param_range = [0.0001, 0.001, .005, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
gs = GridSearchCV(estimator=svm, param_grid=[{'C': param_range, 'gamma': param_range, 'kernel': ['linear','rbf']}], scoring='accuracy', cv=3, n_jobs=-1)

# GridSearch, by default, will return the best model, refit using ALL of the training data.
# Cross Validation, evaluates the returned GridSearch model
cross_val_score(gs, X_train_std, y_train)    
```
```python
gs.fit(X_train_std, y_train)
train_score = gs.score(X_train_std, y_train)
test_score = gs.score(X_test_std, y_test)
print("Train score: {} \nTest score: {}".format(train_score, test_score))
```


{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
