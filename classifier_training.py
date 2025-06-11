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
#from csv_reader_cifar10 import *

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
        
      while True:
        try:
            kernel_choice = int(input("Specify your desired kernel: 0 - RBF, 1 - Linear, 2 - Poly "))
        
            if kernel_choice == 0:
            # rbf:

            #Train:
                svm_rbf = SVC(kernel="rbf", gamma=0.5, C=1.0)
                svm_rbf.fit(X_train, y_train)
               
                print("---SVM with RBF Kernel---")
                prediction_report(svm_rbf, X_train, y_train,X_test, y_test)

        
            elif kernel_choice == 1:
            # linear

            #Train:
                svm_linear = SVC(kernel="linear", C=1.0)
                svm_linear.fit(X_train, y_train)
                
                print("---SVM with Linear Regression Kernel---")
                prediction_report(svm_linear, X_train, y_train,X_test, y_test)

            elif kernel_choice == 2:
            # poly

             #Train:
                svm_poly = SVC(kernel="poly", gamma=0.5, C=1.0)
                svm_poly.fit(X_train, y_train)
                
                print("---SVM with Polynomial Kernel---")
                prediction_report(svm_poly, X_train, y_train,X_test, y_test)

            break
        
        except ValueError:
            print("Please input an integer value!")
            

        
    elif int_choice==1:

        #train Logisitic Regression
        model = LogisticRegression(random_state=16)
        model.fit(X_train, y_train)

        print("---Logisitic Regression---")
        prediction_report(model, X_train, y_train,X_test, y_test)


    elif int_choice==2:
        
        #train Decision Tree
        model=DecisionTreeClassifier()
        model.fit(X_train,y_train)
        
        print("---Decision Tree---")
        prediction_report(model, X_train, y_train,X_test, y_test)
       

    elif int_choice==3:
        #train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        print("---Random Forest---")
        prediction_report(model, X_train, y_train,X_test, y_test)
        
    else: 
        print("Invalid choice - Please choose one of the specified values!")


def prediction_report(model, X_train, y_train,X_test, y_test):
     # Training Set         
        predictions_train_y=model.predict(X_train)
        print("Training Set:")
        print("Accuracy:", accuracy_score(y_train, predictions_train_y))
        print("Precision:", precision_score(y_train, predictions_train_y, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'))  
        print("Recall:", recall_score(y_train, predictions_train_y, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'))
        print("F1 Score:", f1_score(y_train, predictions_train_y, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'))
        
        # Test Set       
        predictions_test_y=model.predict(X_test)
        print("Testing Set:")
        print("Accuracy:", accuracy_score(y_test, predictions_test_y))
        print("Precision:", precision_score(y_test,predictions_test_y, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'))  
        print("Recall:", recall_score(y_test, predictions_test_y, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'))
        print("F1 Score:", f1_score(y_test, predictions_test_y, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'))


if __name__ == '__main__':

  #You can use a different csv file. This is just a sample
  csv_file_path = "imagenet_experiment_results.csv" 
  folder_name = "csv_images"
  img_url_col = "image_link"
  percent = 0.2
  
  download_img(csv_file_path, folder_name, img_url_col)
  
  myPairs, myChecks = csv_reader(csv_file_path)
  
  train_dataset, test_dataset = dataset_split(percent, myPairs)
  '''while True:
        try:
            index = int(input("\nEnter index to display image pair (type -1 to exit): "))
            if index == -1:
                break
            display_img(index, myPairs)
        except ValueError:
            print("Please enter a valid integer.")'''

    # specify what classifier you want
  print("classifiers: 0 - SVM, 1 - Logistic Regression, 2 - Decision Tree, 3 - Random Forest")
  while True:
        # try:
            classifier_choice = int(input("Specify your choice of classifier (type -1 to exit):").strip())
            if classifier_choice == -1:
                break
            elif classifier_choice == 0 or classifier_choice == 1 or classifier_choice == 2 or classifier_choice == 3 :
                classifier_modeler(classifier_choice, train_dataset, test_dataset)
            else:
                print("Please choose one of the specified values!")

        #except ValueError:
         #   print("Please enter a valid integer.")



