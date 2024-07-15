#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix,classification_report,f1_score,precision_score,recall_score,roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score,GridSearchCV


# In[2]:


df = pd.read_csv("churn.csv",index_col=0)


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[11]:


categorical_variables = [col for col in df.columns if col in "0"
or df[col].nunique()<=11 and col not in "Exited"]


# In[12]:


categorical_variables


# In[13]:


numeric_variables = [col for col in df.columns if df[col].dtype!="object"
                    and df[col].nunique()>11
                    and col not in "CustomerId"]


# In[14]:


numeric_variables


# In[15]:


df["Exited"].value_counts()


# In[16]:


churn  = df.loc[df["Exited"]==1]


# In[17]:


not_churn = df.loc[df["Exited"]==0]


# In[18]:


#Categorical Variables
not_churn["Tenure"].value_counts().sort_values()


# In[19]:


churn["Tenure"].value_counts().sort_values()


# In[20]:


not_churn["NumOfProducts"].value_counts().sort_values()


# In[21]:


churn["NumOfProducts"].value_counts().sort_values()


# In[22]:


not_churn["HasCrCard"].value_counts()


# In[23]:


churn["HasCrCard"].value_counts()


# In[24]:


not_churn["IsActiveMember"].value_counts()


# In[25]:


churn["IsActiveMember"].value_counts()


# In[26]:


not_churn.Geography.value_counts().sort_values()


# In[27]:


churn.Geography.value_counts().sort_values()


# In[28]:


not_churn.Gender.value_counts()


# In[30]:


churn.Gender.value_counts()


# In[31]:


not_churn["CreditScore"].describe()


# In[34]:


pyplot.figure(figsize=(10,8))
pyplot.xlabel('CreditScore')
pyplot.hist(not_churn["CreditScore"],bins=15,alpha=0.7,label='Not Churn')
pyplot.legend(loc='upper right')
pyplot.show()


# In[35]:


churn["CreditScore"].describe()


# In[36]:


pyplot.figure(figsize=(10,8))
pyplot.xlabel('CreditScore')
pyplot.hist(churn["CreditScore"],bins=15,alpha=0.8,label='Churn')
pyplot.legend(loc='upper right')
pyplot.show()


# In[37]:


sns.catplot("Exited","CreditScore",data=df)


# In[38]:


not_churn["Age"].describe()


# In[39]:


pyplot.figure(figsize=(10,8))
pyplot.xlabel('Age')
pyplot.hist(not_churn["Age"],bins=15,alpha=0.7,label='Not Churn')
pyplot.legend(loc='upper right')
pyplot.show()


# In[40]:


churn["Age"].describe()


# In[41]:


pyplot.figure(figsize=(10,8))
pyplot.xlabel('Age')
pyplot.hist(churn["Age"],bins=15,alpha=0.7,label='Churn')
pyplot.legend(loc='upper right')
pyplot.show()


# In[42]:


sns.catplot("Exited","Age", data = df)


# In[43]:


not_churn["Balance"].describe()


# In[44]:


pyplot.figure(figsize=(10,8))
pyplot.xlabel('Balance')
pyplot.hist(not_churn["Balance"],bins=15,alpha=0.7, label='Not Churn')
pyplot.legend(loc='upper right')
pyplot.show()


# In[45]:


churn["Balance"].describe()


# In[46]:


pyplot.figure(figsize=(10,8))
pyplot.xlabel('Balance')
pyplot.hist(churn["Balance"],bins=15,alpha=0.7,label='Churn')
pyplot.legend(loc='upper right')
pyplot.show()


# In[47]:


sns.catplot("Exited","Balance",data=df)


# In[48]:


not_churn["EstimatedSalary"].describe()


# In[49]:


pyplot.figure(figsize=(10,8))
pyplot.xlabel('EstimatedSalary')
pyplot.hist(not_churn["EstimatedSalary"],bins=15,alpha=0.7,label='Not Churn')
pyplot.legend(loc='upper right')
pyplot.show()


# In[50]:


churn["EstimatedSalary"].describe()


# In[52]:


pyplot.figure(figsize=(10,8))
pyplot.xlabel('EstimatedSalary')
pyplot.hist(churn["EstimatedSalary"],bins=15,alpha=0.7,label='churn')
pyplot.legend(loc='upper right')
pyplot.show()


# In[53]:


sns.catplot("Exited","EstimatedSalary",data=df)


# In[55]:


k= 10
cols=df.corr().nlargest(k,'Exited')['Exited'].index
cm= df[cols].corr()
plt.figure(figsize=(10,6))
sns.heatmap(cm,annot=True,cmap='viridis')


# In[56]:


df.isnull().sum()


# In[60]:


def outlier_thresholds(dataframe,variable,low_quantile=0.05,up_quantile=0.95):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range=quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


# In[65]:


def has_outliers(dataframe,numeric_columns,plot=False):
    for col in numeric_columns:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col,"  :  ", number_of_outliers, "outliers")
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()


# In[66]:


for var in numeric_variables:
    print(var,"has", has_outliers(df,[var]),"Outliers")


# In[67]:


df["NewTenure"] = df["Tenure"]/df["Age"]
df["NewCreditScore"] = pd.qcut(df['CreditScore'],6,labels=[1,2,3,4,5,6])
df["NewAgeScore"] = pd.qcut(df['Age'],8,labels=[1,2,3,4,5,6,7,8])
df["NewBalanceScore"] = pd.qcut(df['Balance'].rank(method = "first"), 5, labels =[1,2,3,4,5])
df["NewEstSalaryScore"] = pd.qcut(df['EstimatedSalary'], 10,labels =[1,2,3,4,5,6,7,8,9,10])


# In[68]:


df.head()


# In[69]:


list = ["Gender","Geography"]
df = pd.get_dummies(df,columns = list,drop_first = True)


# In[70]:


df.head()


# In[71]:


df = df.drop(["CustomerId","Surname"],axis = 1)


# In[76]:


def robust_scaler(variable):
    var_median = variable.median()
    quartile1 = variable.quantile(0.25)
    quartile3 = variable.quantile(0.75)
    interquantile_range = quartile3 - quartile1
    if int(interquantile_range) == 0:
        quartile1 = variable.quantile(0.05)
        quartile3 = variable.quantile(0.95)
        interquantile_range = quartile3 - quartile1
        if int(interquantile_range) == 0:
            quartile1 = variable.quantile(0.01)
            quartile3 = variable.quantile(0.99)
            interquantile_range = quartile3 - quartile1
            z = (variable - var_median) / interquantile_range
            return round(z, 3)

        z = (variable - var_median) / interquantile_range
        return round(z, 3)
    else:
        z = (variable - var_median) / interquantile_range
    return round(z, 3)


# In[77]:


new_cols_ohe = ["Gender_Male","Geography_Germany","Geography_Spain"]
like_num = [col for col in df.columns if df[col].dtypes != 'O' and len(df[col].value_counts()) <= 10]
cols_need_scale = [col for col in df.columns if col not in new_cols_ohe
                   and col not in "Exited"
                   and col not in like_num]

for col in cols_need_scale:
    df[col] = robust_scaler(df[col])


# In[78]:


df.head()


# In[84]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
X = df.drop("Exited",axis =1)
y = df["Exited"]
X_train, X_test, y_train, Y_test = train_test_split(X,y,test_size =0.20,random_state =12345)
models = [('LR', LogisticRegression(random_state = 123456)),
         ('KNN',KNeighborsClassifier()),
         ('CART', DecisionTreeClassifier(random_state=123456)),
         ('RF',RandomForestClassifier(random_state=123456)),
         ('SVR', SVC(gamma='auto',random_state = 123456)),
         ('GB',GradientBoostingClassifier(random_state=12345)),
         ]
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits =10)
    cv_results = cross_val_score(model,X,y,cv=kfold)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name,cv_results.mean(),cv_results.std())
    print(msg)


# In[87]:


model_GB = GradientBoostingClassifier(random_state=12345)
model_GB.fit(X_train, y_train)
y_pred = model_GB.predict(X_test)
conf_mat = confusion_matrix(y_pred,Y_test)
conf_mat


# In[88]:


print("True Positive:", conf_mat[1, 1])
print("True Negative:", conf_mat[0, 0])
print("False Positive:", conf_mat[0, 1])
print("False Negative:", conf_mat[1, 0])


# In[89]:


print(classification_report(model_GB.predict(X_test),Y_test))


# In[94]:


def generate_auc_roc_curve(clf, X_test):
    y_pred_proba = clf.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba)
    auc = roc_auc_score(Y_test, y_pred_proba)
    plt.plot(fpr, tpr, label= "AUC ROC Curve with Area Under the curve = "+str(auc))
    plt.legend(loc=4)
    plt.show()
    pass


# In[95]:


generate_auc_roc_curve(model_GB, X_test)


# In[97]:


gbm_model = GradientBoostingClassifier()
gbm_params = {'learning_rate': 0.1,'max_depth':3,'n_estimators':200,'subsample':1}
gbm_tuned = GradientBoostingClassifier(**gbm_params).fit(X,y)


# In[98]:


model_name = "GB"
model = gbm_tuned

kfold = KFold(n_splits=10, random_state=123456, shuffle=True)

cv_results = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
msg = "%s: %f (%f)" % (model_name, cv_results.mean(), cv_results.std())
print(msg)


# In[ ]:





# In[ ]:




