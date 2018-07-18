import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os
os.chdir("C:\\Users\\rgoulika\\Downloads\\Datasets\\Datasets")
student_data=pd.read_csv("studentInfo.csv")
student_data.isna().any()
from sklearn.model_selection import train_test_split
X=student_data.loc[:,['code_module','gender','region','highest_education','imd_band','age_band','num_of_prev_attempts','studied_credits','disability']]
X.code_module.unique()
X.region.unique()
X.imd_band.unique()
X.age_band.unique()
X.disability.unique()
fil=X.imd_band.isna()
X[fil]
X.mode()
mode_imd_band=str(X['imd_band'].mode())
print(mode_imd_band[1])
X['imd_band'].fillna("20-30%",inplace=True)
X.head(100)
X.groupby('imd_band')['age_band'].value_counts()
label_encoder=LabelEncoder()
Y=student_data.iloc[:,-1].values
Y=label_encoder.fit_transform(Y)
X["code_module"]=label_encoder.fit_transform(X["code_module"])
X["gender"]=label_encoder.fit_transform(X["gender"])
X["region"]=label_encoder.fit_transform(X["region"])
X["highest_education"]=label_encoder.fit_transform(X["highest_education"])
X["imd_band"]=label_encoder.fit_transform(X["imd_band"])
X["age_band"]=label_encoder.fit_transform(X["age_band"])
X["disability"]=label_encoder.fit_transform(X["disability"])
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
log_reg=LogisticRegression()
student_data['final_result'].hist()
log_reg.fit(x_train,y_train)
pred=log_reg.predict(x_test)
log_reg.score(x_test,y_test)
from sklearn.metrics import confusion_matrix
cm_df=pd.DataFrame(confusion_matrix(y_test,pred).T)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
RF_model=RandomForestClassifier(max_depth=5,n_estimators=500,oob_score=True)
RF_model.fit(x_train,y_train)
RF_model.score(x_test,y_test)
scale_values=StandardScaler()
from sklearn.ensemble import GradientBoostingClassifier
xgb_model.score(x_test,y_test)
x_train=scale_values.fit_transform(x_train)
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the Third hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
#classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
classifier.fit(x_train, y_train, batch_size = 10, epochs = 30)
import seaborn as sns
sns.countplot(data=student_data,x="final_result",hue="gender")
sns.countplot(data=student_data,x="final_result",hue="disability")
sns.countplot(data=student_data,x="final_result",hue="code_module")
student_data.groupby('final_result')['code_module'].describe()
student_data.groupby('final_result')['disability'].describe()
sns.countplot(data=student_data,x="final_result",hue="imd_band")
sns.countplot(data=student_data,x="final_result",hue="age_band")
x_train,x_test,y_train,y_test=train_test_split(new_X,Y,test_size=0.2)
xgb_model.fit(x_train,y_train)
xgb_model.score(x_test,y_test)
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
svm=LinearSVC()
rfe=RFE(svm,3)
rfe=rfe.fit(X,Y)
print(rfe.support_)
print(rfe.ranking_)
print(X.info())
new_X_FS=X.loc[:,['age_band','num_of_prev_attempts','disability']]
new_X_FS=scale_values.fit_transform(new_X_FS)
x_train,x_test,y_train,y_test=train_test_split(new_X_FS,Y,test_size=0.2)
xgb_model.fit(x_train,y_train)
xgb_model.score(x_train,y_train)
