#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: shiqi
"""
import os
os.chdir('/Users/shiqi/Downloads/Data/')
import pandas as pd
import numpy as np

data = pd.read_csv('Data.csv')
data.info()
data.columns

head = data.head()
# Q1. What percent of customers returned after the first visit?
data['firstVisitDate'].describe()
data['secondVisitDate'].describe()
data['wordCount'].describe()
data['totalVisits'].describe()
data['totalVisits'].unique()

data['customerID'].nunique()
data = data.sort_values(by=['customerID'])
df = data.groupby(['customerID'])['firstVisitDate'].count()
df2 = data.groupby(['customerID'])['secondVisitDate'].count()
dt = data[data['customerID'] == 1381]
# 
unique_first_visitor = data['customerID'].nunique()

second_visit = data[['customerID','secondVisitDate']]
second_visit = second_visit.drop_duplicates()
second_visit['secondVisitDate'].describe()
unique_second_visitor = 1112

percent_returned = unique_second_visitor / unique_first_visitor

# Q2. What are the top three best-performing stories in each section, by pageviews?
df3 = data.groupby(['section', 'headline'])['customerID'].count()
df3 = df3.to_frame()
df3 = df3.rename(columns={"customerID": "PageView"})
df3 = df3.sort_values(by=['section','PageView'],ascending=False)
top_by_section = df3.groupby(['section']).head(3)

# Q3. Based on this data, would you choose to promote a Tech story or a Markets story on social media? Why?
top_by_section = top_by_section.reset_index()
q3 = top_by_section[(top_by_section['section'] == 'WSJ_Tech') | (top_by_section['section'] == 'WSJ_Markets')]

# consider second visit
df4 = data.groupby(['section'])[['customerID','secondVisitDate']].count()
df4['return_rate'] = df4['secondVisitDate']/df4['customerID']
# calculate the significant level
# two sample t-test
group1 = np.zeros(992) 
group1[:118] = 1
group2 = np.zeros(704)
group2[:74] = 1
import scipy.stats as stats
stats.ttest_ind(a=group1, b=group2, equal_var=False)
# recommend Market though these two are not significantly different

# Q4. Create a visualization exploring the relationship between any of the content characteristics 
# (such as section, author, keywords etc...) and returning visitors.
#### group by author
# to make it consistent in author name
data['author'] = data['author'].str.replace('+',' ')
# multiple authors will be splited
df5 = data.groupby(['author'])[['customerID','secondVisitDate']].count()
df5 = df5.reset_index()
df5 = df5.rename(columns={"customerID": "PageView","secondVisitDate":"SecondVisit"})

df5['author'][8].split("|").unlist()
len(df5['author'][1].split("|"))

author = df5['author'].tolist()
page_view = df5['PageView'].tolist()
second_view = df5['SecondVisit'].tolist()
name = []
first = []
second = []
    
for a,p,s in zip(author,page_view,second_view):
    au = a.split("|")
    pa = [p] * len(au)
    se = [s] * len(au)
    name.append(au)
    first.append(pa)
    second.append(se)

name = [item for sublist in name for item in sublist]
first = [item for sublist in first for item in sublist]
second = [item for sublist in second for item in sublist]

df6 = {'author':name,
        'PageView':first,
        'SecondVisit': second}
 
# Create DataFrame
df6 = pd.DataFrame(df6)
df7 = df6.groupby(['author'])[['PageView','SecondVisit']].sum()
df7['return_rate'] = df7['SecondVisit']/df7['PageView']
df7 = df7.sort_values(by=['PageView','return_rate'],ascending=False)
df7['return_rate'].describe()

#### group by keywords
data['topicKeywords'].describe()
df8 = data.groupby(['topicKeywords'])[['customerID','secondVisitDate']].count()
df8 = df8.reset_index()
df8 = df8.rename(columns={"customerID": "PageView","secondVisitDate":"SecondVisit"})
df8['return_rate'] = df8['SecondVisit']/df8['PageView']
df8 = df8.sort_values(by=['PageView','return_rate'],ascending=False)

# split the keywords

keyword = df8['topicKeywords'].tolist()
page_view = df8['PageView'].tolist()
second_view = df8['SecondVisit'].tolist()
word = []
first = []
second = []
    
for a,p,s in zip(keyword,page_view,second_view):
    au = a.split("|")
    au = [x.lower() for x in au] # convert to lowercase
    pa = [p] * len(au)
    se = [s] * len(au)
    word.append(au)
    first.append(pa)
    second.append(se)

word = [item for sublist in word for item in sublist]
first = [item for sublist in first for item in sublist]
second = [item for sublist in second for item in sublist]

df9 = {'keyword':word,
        'PageView':first,
        'SecondVisit': second}

# Create DataFrame
df9 = pd.DataFrame(df9)
df10 = df9.groupby(['keyword'])[['PageView','SecondVisit']].sum()
df10['return_rate'] = df10['SecondVisit']/df10['PageView']
df10 = df10.sort_values(by=['PageView','return_rate'],ascending=False)
df10['return_rate'].describe()

from wordcloud import WordCloud
import matplotlib.pyplot as plt
# trim the data - get rid of minor frequencies
df11 = df10[df10['PageView'] > 100]
df11 = df11.drop(df11.index[2]) # delete the blank row
# create a dictionary for keywords and return rate
df11 = df11.reset_index()
keys = df11['keyword'].tolist()
values = df11['return_rate'].tolist()
d = {keys[i]: values[i] for i in range(len(keys))}
  
wordcloud = WordCloud(max_words=50, background_color="white").generate_from_frequencies(d)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# Q5 Other ideas
# Look at totalVisits
data['totalVisits'].describe()
visit_freq_2 = data[data['totalVisits'] > 1]
visit_freq_2['totalVisits'].hist()
visit_freq_2['totalVisits'].describe()
visit_freq_2['secondVisitDate'].isnull().sum() # check if there is no returned user

dt = visit_freq_2.groupby('section')['firstVisitDate'].count()
# range and group comparison
# 2-5 5-10 10-15 15-20 20+

range_result = []
range_list = [list(range(2,6)),list(range(6,10)),list(range(10,20)),list(range(20,100))]
xtick = ['2-5','5-9','10-19','>20']
for r ,xt in zip(range_list,xtick):
    subset = data[data['totalVisits'].isin(r)]
    dt = subset.groupby('section')['firstVisitDate'].count()
    #dt = dt.sort_values(ascending=False)
    dt = dt/dt.sum()
    temp = pd.DataFrame(columns = dt.index)
    temp.loc[xt] = dt.values
    range_result.append(temp)
colors = plt.cm.GnBu(np.linspace(0, 1, 6))
section_distribution = pd.concat(range_result)
fig,ax = plt.subplots(figsize=(10,7))
section_distribution.plot(kind='bar',stacked=True,ax=ax,color=colors,rot=0)
ax.legend(loc='upper right',bbox_to_anchor=(1.2, 1.0))
ax.set_ylabel("FirstView Distribution of Returned Visitors")

test = data[(data['totalVisits'] > 20) & (data['totalVisits'] < 21)]
test = data[data['totalVisits'] > 20]

















