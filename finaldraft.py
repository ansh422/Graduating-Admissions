import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
dataset= pd.read_csv("Admission_Predict.csv")
# dataset contains the following columns
# Serial no, {GRE Score, TOEFL Score, Univeristy Rating, SOP, LOR, CGPA, Research}(x), Chance of Admit(y)
# to start working, first we need to divide into X and Y
# print(dataset.shape) --> 400x9
# First, we need to drop the Serial Number from the dataset

dataset=dataset.drop(['Serial No.'],axis=1)

# after dropping the shape of dataset would be 400x8 
# now we need to distribution of each feature which requires uses matplot lib library.
# But we can also use seaborn library 

### DATA VISUALIZATION BEGINS ###

plt.figure(1)
plt.title("Distributions")
plt.subplot(3,2,1)
plt.hist(dataset['GRE Score'],color='blue')
plt.xlabel('GRE Score')
plt.ylabel('Frequency')

plt.subplot(3,2,2)
plt.hist(dataset['TOEFL Score'],color='red')
plt.xlabel('TOEFL Score')
plt.ylabel('Frequency')

plt.subplot(3,2,3)
plt.hist(dataset['University Rating'],color='yellow')
plt.xlabel('University Rating')
plt.ylabel('Frequency')

plt.subplot(3,2,4)
plt.hist(dataset['SOP'],color='orange')
plt.xlabel('SOP')
plt.ylabel('Frequency')
plt.subplot(3,2,5)

plt.subplot(3,2,6)
plt.hist(dataset['CGPA'],color='blue')
plt.xlabel('CGPA')
plt.ylabel('Frequency')
plt.show()

# after seeing the distribution, we can infer that there are variety of students applied for admission
# then, we need to check if there are features that may relate to each other
# to check we plot the regression line between two and check how much they relate to each other
# this can be easily plotted with seaborn library

plt.figure(2)
fig=sns.regplot(x='GRE Score',y='TOEFL Score',data=dataset)
plt.title("GRE Score vs TOEFL Score")
plt.show()

# From figure 2, we can see that the candidates which have more GRE score have usually, 
# more TOEFL Score, and that is true, because both focusses on English

plt.figure(3)
fig=sns.regplot(x='GRE Score',y='CGPA',data=dataset)
plt.title("GRE Score vs CGPA")
plt.show()

# From figure 3, we can see that the candidates which have more CGPA have usually,
# more GRE Score, which is true, because they are hardworking, although there are exceptions

plt.figure(4)
fig = sns.lmplot(x="CGPA", y="LOR ", data=dataset, hue="Research")
plt.title("GRE Score vs CGPA")
plt.show()

# Letter of Recommendation(LOR) is not related to a person's academic excellance but
# Having Research experience gives a better LOR. Similarly, LOR is not related with GRE
# or TOEFL Score

# We can similarly find how a feature is related to another

# Now, we need to correlation among the features first. The best way is to use heatmap
# from the seaborn library to correlate between variables

#corr=dataset.corr()

#colormap = sns.diverging_palette(220, 10, as_cmap=True)
#sns.heatmap(corr,cmap=colormap,xticklabels=corr.columns,yticklabels=corr.columns,annot=True)
#plt.show()

### DATA VISUALIZATION FINISHES ###

# from the heatmap, we can infer that Chance of Admit is mainly correlated to,
# GRE Score, TOEFL Score, CGPA, University Ranking

# Thus, we calculate Standard Deviation of GRE Score, TOEFL Score and CGPA as 
# it is easily understandable that more positive SD means more chances of admit
# i.e, vallue of feature is to be normalized.

gre_avg=dataset['GRE Score'].mean()
gre_d=dataset['GRE Score'].std()
diff=dataset['GRE Score']-gre_avg
dataset['GRE_SD']=diff/gre_d

toefl_avg=dataset['TOEFL Score'].mean()
toefl_std=dataset['TOEFL Score'].std()
dif_toefl=dataset['TOEFL Score']-toefl_avg
dataset['TOEFL_SD']=dif_toefl/toefl_std

cgpa_avg=dataset['CGPA'].mean()
cgpa_std=dataset['CGPA'].std()
dif_cgpa=dataset['CGPA']-cgpa_avg
dataset['CGPA_SD']=dif_cgpa/cgpa_std

# Now, we have examined all the features in this project. Now, we need to
# start with a basic model first, and we will use Logistic Regression
# Before applying it, we need to split the dataset between Training and Test datasets
# where generally Training set : Test set = 70:30


df=dataset.values
X= df[:,0:11]
y=df[:,10]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 42)


#Now, it's finally time to apply our first model

lr=LogisticRegression(random_state=3,solver='liblinear',max_iter=100)

# Before fitting this model to the training , we need to first make 
# a label ( 1 if chance of Admit is >0.8 else 0) to both y_train and y_test before 
# logistics Regression is not for 'Continuous' Data. If you try to fot, without labelling
# you will get an error

y_train_label= [1 if value>=0.8 else 0 for value in y_train]
y_test_label=[1 if value>=0.8 else 0 for value in y_test]

lr.fit(X_train,y_train_label)

# After fitting this model, we need to calculate it's  report
# which is done in classification report in sklearn

y_pred=lr.predict(X_test)

print(classification_report(y_test_label,y_pred))

# to see how your model worked calcualte it's  Accuracy Score, Precision Score
# f1- Score, Recall score. Calculate and find out.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("Accuracy Score = ",accuracy_score(y_test_label, y_pred))
print("precision_score: ", precision_score(y_test_label,y_pred))
print("recall_score: ", recall_score(y_test_label,y_pred))
print("f1_score: ",f1_score(y_test_label,y_pred))
