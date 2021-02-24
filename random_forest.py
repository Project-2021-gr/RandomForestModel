import pandas as pd
import imblearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#Settings for displaying dataframe
pd.set_option('display.max_rows', 1000)
pd.set_option('max_columns', None)

#Reads the csv that contains the data and puts them in a dataframe
#The data in the csv are separated with ; delimiter
dataset = pd.read_csv("BankChurners.csv", delimiter = ";")

#Y gets the first column (Attrition Flag) of the dataset and X gets the rest columns
#X will be used for predicting the value of Y in the future
Y = dataset.iloc[:, 1]
X = dataset.iloc[:, 2:]

#Use Label Encoder to encode to int values columns that contrain characters
le1 = LabelEncoder()

#Encodes Y (Attrition Flag column)
Y = le1.fit_transform(Y)

#Encodes X (Gender column)
X['Gender'] = le1.fit_transform((X['Gender']))

#One hot encoding using pandas dummies for categorical data
X = pd.get_dummies(X, columns = ['Dependent_count', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'], drop_first = True)

#Splits X, Y in train and test sets
#Test set's size is 20% of the original dataset
#Random_state is a seed used in the random shuffling
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Scales certain columns using StandardScaler
sc = StandardScaler()

#Scales X_Train using variance and mean
X_train = sc.fit_transform(X_train)

#Scales X_Test using variance and mean that was calculated before so that the model is unbiased
X_test = sc.transform(X_test)

#Oersampling the training set in order to reduce overfitting as one of the two classes of the datasets consists of more than 90% of all the values
#First we use SMOTE (Synthetic Minority Oversampling Technique) on the training set in order to artifially produce data for the minority class.
#Sampling_strategy is 30% so that the minory class has 30% of the size of the majority class
#Random_state is a seed used in the random oversampling
oversample = imblearn.over_sampling.SMOTE(sampling_strategy = 0.3, random_state = 0)

#After smote we oversample the minority class to equal size with the majority class in order to make sure we train the model equally for both classes
oversample2 = imblearn.over_sampling.RandomOverSampler(sampling_strategy = 'minority', random_state = 0)

#Fit and apply
X_train, Y_train = oversample.fit_resample(X_train, Y_train)

#Fit and apply
X_train, Y_train = oversample2.fit_resample(X_train, Y_train)

#Creates the Random Forest Model
#n_estimators = 10 means 10 decision trees
#Entropy criterion for split is used
#Random_state is the seed used in the random construction of the decision trees
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

#Trains the model
classifier.fit(X_train, Y_train)

#Evaluates the results using Cross Validation
#estimator is the object that fits the data
#cv determines the split strategy
results = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)

#Prints the results for the 10 decision trees
print(results)

#Prints the mean score of the results
print("Mean score : {:.2f} %".format(results.mean() * 100))

#Prints the standard deviation of the results
print("Standard deviation: {:.2f} %".format(results.std() * 100))
