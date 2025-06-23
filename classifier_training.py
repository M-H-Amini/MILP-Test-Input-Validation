import pandas as pd
import numpy as np
from sklearn import tree

# Modelling

#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
#from sklearn.model_selection import RandomizedSearchCV, train_test_split 
#from scipy.stats import randint

#Logistic Regression
from sklearn.linear_model import LogisticRegression

# Decision Tree
from sklearn.tree import DecisionTreeClassifier 

# Import functions from other files
from csv_reader import *
#from csv_reader_cifar10 import *


#SVM with kernel (try for all three)
#from sklearn import svm
#from sklearn import datasets
from sklearn.svm import SVC

#Predictions:
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def classifier_modeler(int_choice, kernel_choice, ds_train, ds_test): #classifier_choice
    
    #classifiers: 0 - SVM, 1 - Logistic Regression, 2 - Decision Tree, 3 - Random Forest

    """X_train, y_train = compute_metrics_on_dataset(ds_train, ds_folder='csv_images')
    X_test, y_test = compute_metrics_on_dataset(ds_test, ds_folder='csv_images')"""

    print("Preprocessing training data...")
    X_train, y_train = compute_metrics_on_dataset(ds_train, ds_folder='csv_images')
    print("Preprocessing on training done!")

    print("Preprocessing testing data...")
    X_test, y_test = compute_metrics_on_dataset(ds_test, ds_folder='csv_images') #cifar_10_images
    print("Preprocessing on testing done!")

    model = model_training(int_choice, kernel_choice, X_train, y_train)
    if model is None:
        print("Model training failed.")
        return
    prediction_report(model, X_train, y_train,X_test, y_test)

def model_training(choice, kernel_choice, X_train, y_train):
    if choice == 0:
        
        """  while True:
            try: 
                print("About to ask for kernel choice...")
                kernel_choice = int(input("Specify your desired kernel: 0 - RBF, 1 - Linear, 2 - Poly "))
                if kernel_choice in [0,1,2]:
                    break
            except ValueError:
                print("Please enter a valid integer (0, 1, or 2).")"""
        if kernel_choice == 0:
            # rbf:W
            #Train:
            model = SVC(kernel="rbf", gamma=0.5, C=1.0)
            kernel_name = "RBF Kernel"
            
        
        elif kernel_choice == 1:
        # linear
            #Train:
            model = SVC(kernel="linear", C=1.0)
            kernel_name = "Linear Kernel"
        elif kernel_choice == 2:
            # poly
             #Train:
            model = SVC(kernel="poly", gamma=0.5, C=1.0)
            kernel_name = "Polynomial Kernel"
            
        model.fit(X_train, y_train)
        print("---SVM with", kernel_name,"---") 
                  
    elif choice == 1:
        #train Logisitic Regression
        model = LogisticRegression(random_state=16)
        model.fit(X_train, y_train)
        print("---Logistic Regression---")

    elif choice == 2:
        #train Decision Tree
        model=DecisionTreeClassifier()
        model.fit(X_train,y_train)
        print("---Decision Tree---")

    elif choice == 3:
        #train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("---Random Forest---")

    else:
        print("Invalid choice - Please choose one of the specified values!")
        return None
    
    return model
    


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
  #csv_file_path = "cifar10_experiment_results.csv" 
  #folder_name = "cifar_10_images"
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
  looper = True
  kernel_choice = 0
  while looper == True:
        try:
            classifier_choice = int(input("Specify your choice of classifier (type -1 to exit):").strip())
            if classifier_choice == -1:
                looper = False
            elif classifier_choice == 0 or classifier_choice == 1 or classifier_choice == 2 or classifier_choice == 3 :
                looper = False
                if classifier_choice == 0:
                    while True:
                        try:
                            print("Specify your desired kernel: 0 - RBF, 1 - Linear, 2 - Poly ")
                            kernel_choice = int(input("Specify your choice of kernel (type -1 to exit):").strip())
                            if classifier_choice == -1:
                                break
                            break
                        except ValueError:
                            print("Please input one of the specified integers")
                classifier_modeler(classifier_choice, kernel_choice, train_dataset, test_dataset)
            else:
                print("Please choose one of the specified values!")
        except ValueError:
            print("Please input one of the specified integers")

        #except ValueError:
         #   print("Please enter a valid integer.")


