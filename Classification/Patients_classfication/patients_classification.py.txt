import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
#from sklearn. import 
from sklearn.model_selection import GridSearchCV
import seaborn as sns

df=pd.read_csv('column_2C_weka.csv')

df['class'].replace({'Abnormal':1,'Normal':0},inplace=True)

print(df['class'].value_counts())

print(df.corr())

sns.heatmap(df.corr(),annot=True)

X=df.drop('class',axis=1)
y=df['class']

from scipy.stats import ttest_ind

sns.pairplot(df,hue='class',height=3)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.neighbors import KNeighborsClassifier


knn=KNeighborsClassifier()
param={'n_neighbors':np.arange(1,50),'weights':['uniform','distance']}
GS=GridSearchCV(knn,param,cv=3,scoring='roc_auc')
GS.fit(X,y)

print(GS.best_estimator_)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

lr=LogisticRegression()
knn=GS.best_estimator_
knn1=KNeighborsClassifier(weights='distance',n_neighbors=14)
dtc=DecisionTreeClassifier(criterion='entropy',random_state=0)
dtc_reg=DecisionTreeClassifier(criterion='entropy',random_state=0,max_depth=5)
rf=RandomForestClassifier(n_estimators=9,random_state=0)
models=[]
models.append(('knn',knn))
models.append(('knn1',knn1))
models.append(('lr',lr))
models.append(('dtc',dtc))
models.append(('dtc_reg',dtc_reg))
models.append(('rf',rf))

from sklearn.model_selection import cross_val_score,KFold

results=[]
names=[]
for name, model in models:
    kfold=KFold(shuffle=True,n_splits=3,random_state=0)
    cv_results=cross_val_score(model,X,y,cv=kfold,scoring='roc_auc')
    results.append(cv_results)
    names.append(name)
    #print(cv_results)
    print("%s:  %f (%f)" %(name,np.mean(cv_results),np.var(cv_results,ddof=1)))



