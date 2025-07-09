import pandas as pd
import numpy as np
from sklearn import tree
import tqdm

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
#from csv_reader import *
from mh_optimize import *
from csv_reader_cifar10 import *


#SVM with kernel (try for all three)
#from sklearn import svm
#from sklearn import datasets
from sklearn.svm import SVC

#Predictions:
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def classifier_modeler(model_name, ds_train, ds_test): #classifier_choice
    
    #classifiers: 0 - SVM, 1 - Logistic Regression, 2 - Decision Tree, 3 - Random Forest


    """X_train, y_train = compute_metrics_on_dataset(ds_train, ds_folder='csv_images')
    X_test, y_test = compute_metrics_on_dataset(ds_test, ds_folder='csv_images')"""

    print("Preprocessing training data...")
    #X_train, y_train = compute_metrics_on_dataset(ds_train, ds_folder='csv_images')
    X_train, y_train = compute_metrics_on_dataset(ds_train, ds_folder='cifar_10_images') 
    print("Preprocessing on training done!")
    #print("Total number of unique pairs for Training Set:", totalUnique)

    print("Preprocessing testing data...")
    #X_test, y_test = compute_metrics_on_dataset(ds_test, ds_folder='csv_images') #cifar_10_images
    X_test, y_test = compute_metrics_on_dataset(ds_test, ds_folder='cifar_10_images') #cifar_10_images
    print("Preprocessing on testing done!")
    #print("Total number of unique pairs for Test Set:", totalUnique)

    model = model_training(model_name, X_train, y_train)
    if model is None:
        print("Model training failed.")
        return
    prediction_report(model, X_train, y_train,X_test, y_test)

def model_training(model_name, X_train, y_train): #get rid of total unique later


    if "svm" in model_name:
        if model_name == "svm_rbf":
            model = SVC(kernel="rbf",probability=True, gamma=0.5, C=1.0)
            kernel_name = "RBF Kernel"
        elif model_name == "svm_linear":
            model = SVC(kernel="linear",probability=True, C=1.0)
            kernel_name = "Linear Kernel"
        elif model_name == "svm_poly":
            model = SVC(kernel="poly",probability=True, degree=3)
            kernel_name = "Polynomial Kernel"
        model.fit(X_train, y_train)
        print("---SVM with", kernel_name,"---") 
    elif model_name == "logistic_regression":
        model = LogisticRegression(random_state=16)
        model.fit(X_train, y_train)
        print("---Logistic Regression---")
    elif model_name == "decision_tree":
        model=DecisionTreeClassifier()
        model.fit(X_train,y_train)
        print("---Decision Tree---")
    elif model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("---Random Forest---")
    else:
        print("Error! Please double check your inputed models.")
    
    return model
    
    '''if choice == 0:
        
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
            model = SVC(kernel="rbf",probability=True, gamma=0.5, C=1.0)
            kernel_name = "RBF Kernel"
        
        elif kernel_choice == 1:
        # linear
            #Train:
            model = SVC(kernel="linear",probability=True, C=1.0)
            kernel_name = "Linear Kernel"
        elif kernel_choice == 2:
            # poly
             #Train:
            model = SVC(kernel="poly",probability=True, degree=3)
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
        return None'''
    
    

def prediction_report(model, X_train, y_train,X_test, y_test): #get rid of total unique
     # Training Set         
        predictions_train_y=model.predict(X_train)
        print("Training Set:")
        print("Total number of unique pairs:", len(y_train))
        print("Accuracy:", accuracy_score(y_train, predictions_train_y))
        print("Precision:", precision_score(y_train, predictions_train_y, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'))  
        print("Recall:", recall_score(y_train, predictions_train_y, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'))
        print("F1 Score:", f1_score(y_train, predictions_train_y, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'))
        
        # Test Set       
        predictions_test_y=model.predict(X_test)
        print("Testing Set:")
        print("Total number of unique pairs:", len(X_test))
        print("Accuracy:", accuracy_score(y_test, predictions_test_y))
        print("Precision:", precision_score(y_test,predictions_test_y, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'))  
        print("Recall:", recall_score(y_test, predictions_test_y, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'))
        print("F1 Score:", f1_score(y_test, predictions_test_y, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'))


def train_and_predict(MODELS, d_t, d_o):


    print("Preprocessing training data...")
    #X_dt, y_dt = compute_metrics_on_dataset(d_t, ds_folder='csv_images')
    X_dt, y_dt = compute_metrics_on_dataset(d_t, ds_folder='cifar_10_images') 
    print("Preprocessing on training done!")
    #print("Total number of unique pairs for Training Set:", totalUnique)

    print("Preprocessing testing data...")
    #X_do, y_do = compute_metrics_on_dataset(d_o, ds_folder='csv_images') #cifar_10_images
    X_do, y_do = compute_metrics_on_dataset(d_o, ds_folder='cifar_10_images') #cifar_10_images
    print("Preprocessing on testing done!")
    #print("Total number of unique pairs for Test Set:", totalUnique)

    total_dat = pd.DataFrame()
    classifier_num = 0
    for model_name in MODELS:
        model = model_training(model_name, X_dt, y_dt)
        if model is None:
            print("Modeltraining failed.")
            return
        dat = populate_def_pred(model,classifier_num, X_do)
        classifier_num+=1
    
        total_dat = pd.concat([total_dat, dat], axis=1)

    y = pd.DataFrame({'y': y_do})
    total_dat = pd.concat([total_dat, y], axis=1)
    total_dat.to_csv('predicted_output_on_do.csv', index = False)
    counters_on_dataframe(total_dat)
    effort, w_list, x_list = mh_optimize(total_dat, 1)

    for i in range(0, len(w_list)):
      for value in total_dat["p_theta_"+str(i)]:
            print(value)
            value_sum = 0
            for j in range(0, len(w_list)):
                value_sum += value*w_list[j]
                
            if value_sum >= 1:
                print("POOR ACCURACY:", value_sum)
            else:
                print("GOOD ACCURACY")
    
    

        

    print('Effort:', effort)
    print('w_list:', w_list)
    #print('x_list:', x_list)
    print("Length: ", len(y_do))


def counters_on_dataframe(dataframe):
    
    cnt_auto = 0
    cnt_manual = 0
    cnt_correct = 0

    # Select p_l_* columns only (regardless of order)
    p_l_cols = dataframe.filter(regex=r'^p_l_')

    # For correct check: p_l_ + y
    p_l_cols_with_y = p_l_cols.copy()
    p_l_cols_with_y['y'] = dataframe['y']

    #print(dataframe.iloc[0])  # or dataframe.loc[0]

    for index, row in p_l_cols.iterrows():
        if (row == row.iloc[0]).all():
            cnt_auto+=1
        else:
            cnt_manual+=1
         
    for index, row in p_l_cols_with_y.iterrows():
        if (row == row.iloc[0]).all():
            cnt_correct+=1
    
    print("cnt_auto", cnt_auto)
    print("cnt_manual", cnt_manual)
    print("cnt_correct", cnt_correct) 

    return cnt_auto, cnt_manual, cnt_correct
def populate_def_pred(model, classifier_num, X_do):
        
        # model is trained on d_t, now it must predict d_o and populate df_pred
        
        print("Meow, we're in the populate_def_pred")

        # get P_theta
        batch_size = 128
        p_theta = []
        for i in tqdm(range(0, len(X_do), batch_size), desc= "Predicting Batches"):
            batch = X_do[i:i+batch_size]
            p_theta.append(model.predict_proba(batch)[:, 1])
        p_theta = np.concatenate(p_theta)

    
        #p_theta = model.predict_proba(X_do)[:,1] # so 0.9 = 90% probability 

        # get P_l
        p_l = model.predict(X_do)
        #p_l = (p_theta >= 0.5).astype(float)
        # Get y => we already have it.

        model_name = type(model).__name__
        

        print("Generating the df_pred file for:", model_name)
        data = {
            'p_l_'+ str(classifier_num) : p_l,
            'p_theta_'+ str(classifier_num) : p_theta
           # 'y' + str(classifier_num) : y_do
        }
    
        dat = pd.DataFrame(data)
       

        return dat
        #def_pred.to_csv('predicted_output_on_do.csv', index = False)
      
    

if __name__ == '__main__':

  #You can use a different csv file. This is just a sample
  #csv_file_path = "imagenet_experiment_results.csv" 
  #folder_name = "csv_images" 
  csv_file_path = "cifar10_experiment_results.csv" 
  folder_name = "cifar_10_images"
  img_url_col = "image_link"
  TRAIN_PERCENT = 0.2
  OPT_SPLIT = 0.5
  MODELS = ['svm_rbf', 'logistic_regression', 'random_forest'] # ['svm_linear','decision_tree']
  
  download_img(csv_file_path, folder_name, img_url_col)
  
  myPairs, myChecks = csv_reader(csv_file_path)

  ds_unique = unique_pair_dataset(myPairs)

  #train_dataset, test_dataset = dataset_split(percent, myPairs)

  #train_dataset, test_dataset = dataset_split(TRAIN_PERCENT, ds_unique)


  # for populating df_pred for linear programming
    #make duplicates of the training and test datasets
  #d_nv, d_to =  train_dataset, test_dataset 
  
  d_rest, d_t = dataset_split(TRAIN_PERCENT, ds_unique)
  
  d_nv, d_o = dataset_split(OPT_SPLIT/(1-TRAIN_PERCENT), d_rest)

  #for model in MODELS:
    #classifier_modeler(classifier_choice, kernel_choice, train_dataset, test_dataset)
  train_and_predict(MODELS,d_t, d_o)
  #train_and_predict(MODELS,d_t, d_nv)
  #train_and_predict(MODELS,d_t, d_t)
  print("length of d_t:", len(d_t))
  print("length of d_rest:", len(d_rest)) 
  print("length of d_nv:", len(d_nv))
  print("length of d_o:", len(d_o))
  #
    # 20, then 80   
    # 1/2, then 50/80





  '''# specify what classifier you want
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
                #classifier_modeler(classifier_choice, kernel_choice, train_dataset, test_dataset)
                train_and_predict(classifier_choice, kernel_choice,d_t, d_o)


                #continue - populate df_pred 
                # test on all classifiers, include them in df_pred ==> suggested to just have the two best-performing classifiers, 
                # but keep others just in case.



            else:
                print("Please choose one of the specified values!")
        except ValueError:
            print("Please input one of the specified integers")
'''
        #except ValueError:
         #   print("Please enter a valid integer.")


