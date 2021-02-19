import pandas as pd
import imblearn

#settings for displaying dataframe
pd.set_option('display.max_rows', 1000)
pd.set_option('max_columns', None)

#reading the dataset
dataset=pd.read_csv("BankChurners.csv",delimiter=";")

#Y gets the first column of the dataset and X gets the rest columns
Y=dataset.iloc[:,1]
X=dataset.iloc[:,2:]

#label encoding binary columns
from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()

Y=le1.fit_transform(Y)
X['Gender']=le1.fit_transform((X['Gender']))

#one hot encoding categorical columns
X=pd.get_dummies(X,columns=['Dependent_count','Education_Level','Marital_Status','Income_Category','Card_Category'],drop_first=True)

#splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split

#create X,Y train and test set. Test sets size is 20% of the original dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

#scaling the values of the training in order to improve results
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#oversampling the training set in order to reduce overfitting as one of the two classes of the datasets consists of more than 90% of all the values this is necessary
#first we use smote(Synthetic Minority Oversampling Technique)on the training set in order to artifially produce data for the minority class this cant be done all the way cause of overfitting
oversample=imblearn.over_sampling.SMOTE(sampling_strategy=0.3,random_state=0)

#after smote the dataset we oversample it in order to make sure we train the model equally for both classes
oversample2=imblearn.over_sampling.RandomOverSampler(sampling_strategy='minority',random_state=0)

X_train,Y_train=oversample.fit_resample(X_train,Y_train)
X_train,Y_train=oversample2.fit_resample(X_train,Y_train)

from sklearn.ensemble import RandomForestClassifier

#creating the model(random forest)
classifier = RandomForestClassifier(
	n_estimators=10,
	criterion = 'entropy',
	random_state = 0)

#training the model
classifier.fit(X_train, Y_train)

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier,X = X_train,y = Y_train,cv = 10)

#print the results
print(accuracies)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))