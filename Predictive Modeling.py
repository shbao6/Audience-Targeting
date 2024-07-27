#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shiqi
"""
import os
os.chdir('/Users/shiqi/Downloads/Data/')
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('Data.csv')
data.info()
data.columns
#### Part B
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

#%% Feature Selection
# word count video count
Predictor_word = data[['wordCount', 'videoCount']].copy()
Predictor_word['wordCount'] = Predictor_word['wordCount'].fillna(Predictor_word['wordCount'].median())
Predictor_word['videoCount'] = Predictor_word['videoCount'].fillna(Predictor_word['videoCount'].median())
Predictor_word['wordCount'] = Predictor_word['wordCount']/data['wordCount'].max()

#dummy variable of sections
sections = data['section'].unique()
Predictor_section = pd.DataFrame(columns=sections,index=data.index,data=0)
for j in data.index:
    section_j = data.loc[j,'section']
    Predictor_section.loc[j,section_j]=1
Predictor_section = Predictor_section.drop(columns=['WSJ_Life']) #avoid colinearity

#dummy variable of authors
authors_all = []
author_split= data['author'].str.split("|")
for a in author_split.index:
    au = author_split[a]
    if isinstance(au,float):
        continue
    authors_all = authors_all+au
authors_all = pd.DataFrame(authors_all,columns=['author']) 
authors_all = authors_all[['author']].value_counts().reset_index()
authors_all.columns = ['author','vcounts']
author_cutoff = 200
authors = authors_all.loc[authors_all['vcounts']>author_cutoff,'author'].values#set of authors to use
Predictor_author = pd.DataFrame(columns=authors,index=data.index,data=0)
for j in data.index:
    author_j = author_split[j]
    if isinstance(author_j,float):
        continue
    cindex = np.isin(authors,author_j)
    Predictor_author.loc[j,cindex]=1


#dummy variable of keywords
keywords_all = []
keywords_split= data['topicKeywords'].str.split("|")
for a in keywords_split.index:
    kw = keywords_split[a]
    if isinstance(kw,float):
        continue
    keywords_all = keywords_all+kw
keywords_all = pd.DataFrame(keywords_all,columns=['keywords']) 
keywords_all = keywords_all[['keywords']].value_counts().reset_index()
keywords_all.columns = ['keywords','vcounts']
keywords_cutoff = 500
keywords = keywords_all.loc[keywords_all['vcounts']>keywords_cutoff,'keywords'].values
keywords = keywords[~(keywords=='')]

Predictor_keywords = pd.DataFrame(columns=keywords,index=data.index,data=0)
for j in data.index:
    keywords_j = keywords_split[j]
    if isinstance(keywords_j,float):
        continue
    cindex = np.isin(keywords,keywords_j)
    Predictor_keywords.loc[j,cindex]=1

#target variable
Y = np.zeros(10000)
Y[~data['secondVisitDate'].isnull().values] = 1

from sklearn.ensemble import RandomForestClassifier
#merge 4 set of predictors
Predictor = Predictor_section.join(Predictor_author)
Predictor = Predictor.join(Predictor_keywords)
Predictor = Predictor.join(Predictor_word)

#%% modeling
#Predictor = Predictor_section.join(Predictor_keywords)
X_train, X_test, y_train, y_test = train_test_split(Predictor, Y, test_size=0.2, random_state=42)#split into training and testing set
from sklearn.model_selection import cross_val_score

#decision tree
tree_depth = [2,3,4,5,6,10]
cv_scores_mean = []
for depth in tree_depth:
    print(depth)
    mdl = DecisionTreeClassifier(max_depth=depth,min_samples_leaf=200,max_features='auto',criterion="entropy",random_state=42)
    cv_scores = cross_val_score(mdl,X_train,y_train,cv=10,scoring='roc_auc')
    cv_scores_mean.append(cv_scores.mean())
mdl =DecisionTreeClassifier(max_depth=tree_depth[np.argmax(cv_scores_mean)],min_samples_leaf=200,max_features='auto',random_state=42)

mdl = mdl.fit(X_train,y_train)
y_pred = mdl.predict_proba(X_test)
print("Accuracy:",metrics.roc_auc_score(y_test, y_pred[:,1]))

from matplotlib import pyplot as plt # plot tree
fig = plt.figure(figsize=(25,20))
from sklearn import tree
tree.plot_tree(mdl, feature_names=Predictor.columns,  proportion=True)

#random forest
tree_depth = [5,10,15,20]
cv_scores_mean = []
for depth in tree_depth:
    print(depth)
    rf = RandomForestClassifier(n_estimators=500,max_depth=depth,min_samples_leaf=10,max_features='log2',random_state=42)
    cv_scores = cross_val_score(rf,X_train,y_train,cv=5,scoring='roc_auc')
    cv_scores_mean.append(cv_scores.mean())
#rf = RandomForestClassifier(n_estimators=500,max_depth=30,min_samples_leaf=10)
rf =RandomForestClassifier(n_estimators=500,max_depth=tree_depth[np.argmax(cv_scores_mean)],min_samples_leaf=10,max_features='log2',random_state=42)
rf = rf.fit(X_train,y_train)
y_pred_rf = rf.predict_proba(X_test)
print("Accuracy:",metrics.roc_auc_score(y_test, y_pred_rf[:,1]))

rf_importance = pd.DataFrame(columns=['Predictor','Importance'])#plot variable importance
rf_importance['Predictor'] = Predictor.columns
rf_importance['Importance'] = rf.feature_importances_
rf_importance = rf_importance.sort_values('Importance',ascending=False)
rf_importance = rf_importance.head(10)
plt.figure(figsize=(4,10))
rf_importance.plot.barh(x='Predictor',y='Importance',ax = plt.gca())
  
# lasso bench mark
from sklearn.linear_model import LogisticRegression
C_set = [0.5,0.2,0.1,0.05]
l1_ratio_set = [0.2,0.5,0.8]
cv_scores_mean = pd.DataFrame(columns=['C','l1_ratio','score'])
for C in C_set:
    for l1_ratio in l1_ratio_set:
        print(C)
        print(l1_ratio)
        lgt = LogisticRegression(penalty='elasticnet',l1_ratio=l1_ratio,C=C,solver='saga',random_state=42)
        cv_scores = cross_val_score(lgt,X_train,y_train,cv=10,scoring='roc_auc')
        temp = pd.DataFrame(columns=['C','l1_ratio','score'])
        temp.C = [C]
        temp.l1_ratio=[l1_ratio]
        temp.score = [cv_scores.mean()]
        cv_scores_mean = pd.concat( (cv_scores_mean,temp),ignore_index=True)

ind_optimal_param = np.argmax(cv_scores_mean.score) #choose hyperparameters
C = cv_scores_mean.loc[ind_optimal_param,'C']
l1_ratio = cv_scores_mean.loc[ind_optimal_param,'l1_ratio']
lgt = LogisticRegression(random_state=42,penalty='elasticnet',l1_ratio=l1_ratio,solver='saga',C=C).fit(X_train, y_train)
y_pred_lgt = lgt.predict_proba(X_test)
print("Accuracy:",metrics.roc_auc_score(y_test, y_pred_lgt[:,1]))#test oos performance

df_coef = pd.DataFrame({'predictor':Predictor.columns}) #plot variable importance
df_coef['Coefficient'] = lgt.coef_.ravel()
df_coef = df_coef[df_coef['Coefficient']>0.0001]
df_coef.plot.bar(x='predictor',y='Coefficient', rot=0)

#%combine rf and logit?
print("Accuracy:",metrics.roc_auc_score(y_test, (y_pred_rf[:,1]+y_pred_lgt[:,1])/2))#test oos performance

#%% predictor sub
sub_predictor = ['wordCount','videoCount','WSJ_Politics']

#decision tree
tree_depth = [2,3,4,5,6,10]
cv_scores_mean = []
for depth in tree_depth:
    print(depth)
    mdl = DecisionTreeClassifier(max_depth=depth,min_samples_leaf=100,max_features='auto',criterion="gini",random_state=42)
    cv_scores = cross_val_score(mdl,X_train[sub_predictor],y_train,cv=10,scoring='roc_auc')
    cv_scores_mean.append(cv_scores.mean())
#mdl = RandomForestClassifier(n_estimators=500,max_depth=30,min_samples_leaf=10)
mdl =DecisionTreeClassifier(max_depth=tree_depth[np.argmax(cv_scores_mean)],min_samples_leaf=100,max_features='auto',criterion="gini",random_state=42)

mdl = mdl.fit(X_train[sub_predictor],y_train)
y_pred = mdl.predict_proba(X_test[sub_predictor])
print("Accuracy:",metrics.roc_auc_score(y_test, y_pred[:,1]))

from matplotlib import pyplot as plt # plot figure
fig = plt.figure(figsize=(25,20))
from sklearn import tree
tree.plot_tree(mdl, feature_names=Predictor[sub_predictor].columns,  proportion=True)


# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
