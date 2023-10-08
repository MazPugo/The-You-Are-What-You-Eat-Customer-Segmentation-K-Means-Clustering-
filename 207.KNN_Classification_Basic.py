#KNN
# the KNN algorithm predicts a class for an unknown data point using the most popular class
#of a number of nearby known data points
#the number of nearby data points used to frm the prediction is denoted by k


#KNN Classification




#linear regression tree basic template

# import required python package


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


#import sample data


my_df = pd.read_csv("data/sample_data_classification.csv")



#split data into input and poutput object

X = my_df.drop(["output"], axis =1)
y = my_df["output"]



#split data into training ans tests sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)


#initiate our model object
clf = KNeighborsClassifier()

#train out model

clf.fit(X_train, y_train)

#access model accuracy

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)



#Euclidean distance--The "straight line" distance, commonly used in KNN

#Feature Scaling is where we force the values  from different columns to exist on the same scale, in order to enhance
# the learning capabilities of the model
#the 2 most common techniques are standarisation and normalisation
#Standarisation rescales data to have a mean of 0 and standard deviation of 1
#Normalisation rescales data so that it exists in a range between 0 and 1
#always use scallin when you use KNN, use normalisation rather than standarisation
#do not go to low with k a rogue outlier could cause incorrect classification
#do not go to high with your value for k, consider class size in the data. Do not use a value of k that willl overhelm smaller classes
#test different values of k, plotting a value of k against model accuracy should provide a view on a sensible value