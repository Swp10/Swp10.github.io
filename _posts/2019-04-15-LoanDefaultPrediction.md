---
title: "Loan Default Prediction"
date: 2019-04-15
tages: [machine learning, data science, data modeling, numpy, pandas, matplotlib, scikit-learn]
excerpt: "Machine Learning, Data Science, Data Modeling, Numpy, Pandas, Matplotlib, Scikit-learn"
gallery:
  - url: /images/Loan.jpg
    image_path: /images/Loan.jpg
    alt: "title image"
    title: "Loan"
---

{% include gallery id="gallery" layout="half" caption="Loan" %}

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

Check out the info(), describe() and head() methods on loan
```python
loan.info()
loan.describe()
loan.head()
```

Check if there is any missing data
```python
loan.isnull().sum()
```

Check the size of the dataset
```python
loan.shape
```




{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
