import pandas as pd
import numpy as np
from sklearn import tree

# Modelling

#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

#Logistic Regression
from sklearn.linear_model import LogisticRegression

# Decision Tree
from sklearn.tree import DecisionTreeClassifier 

# Import functions from other files
from csv_reader import *
from csv_reader_cifar10 import *

#SVM with kernel (try for all three)
from sklearn import svm
from sklearn import datasets
from sklearn.svm import SVC

#Predictions:
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


'''
# Load the Iris dataset
iris = datasets.load_iris()
# We only take the first two
# features for simplicity
X = iris.data[:, :2]
y = iris.target

# Fit the SVM model
model = svm.SVC(kernel='linear')
model.fit(X, y)

# Predict using the SVM model
predictions = model.predict(X)

# Evaluate the predictions
accuracy = model.score(X, y)
print("Accuracy of SVM:", accuracy)
'''



"""#SVM (kernel = rbf, poly and something else)
#Logistic Regression
#Decision Tree
#Random Forest

#Better as numpy array
"""



def classifier_modeler(int_choice, ds_train, ds_test): #classifier_choice
    
    #classifiers: 0 - SVM, 1 - Logistic Regression, 2 - Decision Tree, 3 - Random Forest
    
    
    X_train, y_train = compute_metrics_on_dataset(ds_train, ds_folder='csv_images')
    X_test, y_test = compute_metrics_on_dataset(ds_test, ds_folder='csv_images')


    if int_choice==0:
        #train SVM
        # kernels: linear, polynomial, rbf
        
        kernel_choice = input("Specify your desired kernel: ")
        
        if kernel_choice == 0:
            # rbf:
            svm_rbf = SVC(kernel="rbf", gamma=0.5, C=1.0)
            svm_rbf.fit(X_train, y_train)
            predictions_train_y = model.predict(X_train)
            print("---Logisitic Regression---")
            print("Training Set:")
            print("Accuracy:", accuracy_score(y_train, predictions_train_y))
            print("Precision:", precision_score(y_train, predictions_train_y, average='binary'))  
            print("Recall:", recall_score(y_train, predictions_train_y, average='binary'))
            print("F1 Score:", f1_score(y_train, predictions_train_y, average='binary'))





        pass
    elif int_choice==1:

        #train Logisitic Regression
        model = LogisticRegression(random_state=16)
        model.fit(X_train, y_train)

        # Training Set         

        predictions_train_y = model.predict(X_train)
        print("---Logisitic Regression---")
        print("Training Set:")
        print("Accuracy:", accuracy_score(y_train, predictions_train_y))
        print("Precision:", precision_score(y_train, predictions_train_y, average='binary'))  
        print("Recall:", recall_score(y_train, predictions_train_y, average='binary'))
        print("F1 Score:", f1_score(y_train, predictions_train_y, average='binary'))
        
        # Test Set       

        predictions_test_y = model.predict(X_test)
        print("Testing Set:")
        print("Accuracy:", accuracy_score(y_test, predictions_test_y))
        print("Precision:", precision_score(y_test,predictions_test_y, average='binary'))  
        print("Recall:", recall_score(y_test, predictions_test_y, average='binary'))
        print("F1 Score:", f1_score(y_test, predictions_test_y, average='binary'))



    elif int_choice==2:
        
        #train Decision Tree
        model=DecisionTreeClassifier()
        model.fit(X_train,y_train)

        # Training Set         

        predictions_train_y=model.predict(X_train)
        
        print("---Decision Tree---")
        print("Training Set:")
        print("Accuracy:", accuracy_score(y_train, predictions_train_y))
        print("Precision:", precision_score(y_train, predictions_train_y, average='binary'))  
        print("Recall:", recall_score(y_train, predictions_train_y, average='binary'))
        print("F1 Score:", f1_score(y_train, predictions_train_y, average='binary'))
        
        # Test Set       

        predictions_test_y=model.predict(X_test)
        print("Testing Set:")
        print("Accuracy:", accuracy_score(y_test, predictions_test_y))
        print("Precision:", precision_score(y_test,predictions_test_y, average='binary'))  
        print("Recall:", recall_score(y_test, predictions_test_y, average='binary'))
        print("F1 Score:", f1_score(y_test, predictions_test_y, average='binary'))

    elif int_choice==3:
        #train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Training Set         
        predictions_train_y=model.predict(X_train)
        print("---Random Forest---")
        print("Training Set:")
        print("Accuracy:", accuracy_score(y_train, predictions_train_y))
        print("Precision:", precision_score(y_train, predictions_train_y, average='binary'))  
        print("Recall:", recall_score(y_train, predictions_train_y, average='binary'))
        print("F1 Score:", f1_score(y_train, predictions_train_y, average='binary'))
        
        # Test Set       
        predictions_test_y=model.predict(X_test)
        print("Testing Set:")
        print("Accuracy:", accuracy_score(y_test, predictions_test_y))
        print("Precision:", precision_score(y_test,predictions_test_y, average='binary'))  
        print("Recall:", recall_score(y_test, predictions_test_y, average='binary'))
        print("F1 Score:", f1_score(y_test, predictions_test_y, average='binary'))
    else: 
        print("Invalid choice")

    

'''
music_d=pd.read_csv('music.csv')
X=music_d.drop(columns=['genre'])
y=music_d['genre']


model=DecisionTreeClassifier()
model.fit(X,y)
prediction=model.predict([[21,1],[22,0]])
prediction
'''

"""
 => CALL THIS IN THE CSV_READER FILE
if __name__=='__main__':

    # specify what classifier you want
    print("classifiers: 0 - SVM, 1 - Logistic Regression, 2 - Decision Tree, 3 - Random Forest")
    while True:
        try:
            classifier_choice = int(input("Specify your choice of classifier:"))
            if ((classifier_choice > 3) or (classifier_choice <0)):
                print("Please use one of the given numbers")
            else:
                #call functtion that trains classifiers
                pass

        except ValueError:
            print("Please enter a valid integer.")
        
        
       """ 

"""
#Report
    # Precision
    # Accuracy
    # Recall
    # FI-Score
"""

def classifier_report():
    
    
    pass
