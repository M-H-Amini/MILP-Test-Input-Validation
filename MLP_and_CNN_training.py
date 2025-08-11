import os
import numpy as np
from PIL import Image

#Set our gpu/cuda
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
#from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import classification_report

#for visualizations
import matplotlib.pyplot as plt


""" DOWNLOAD ALL THE IMAGES """

#create folder if it doesn't exist
#folder_name = "MLP_and_CNN_training"

def download_img(folder_name):
    if os.path.exists(folder_name):
      print("Images already downloaded")
    else:
      os.makedirs(folder_name, exist_ok = True)

    #load all of MNIST and split into train and test sets
      (x_train, y_train), (x_test, y_test) = mnist.load_data()
      total_img = np.concatenate((x_train, x_test), axis = 0)
      total_labels= np.concatenate((y_train, y_test), axis = 0) 

      for i, (img, label) in enumerate(zip(total_img, total_labels)):
        label_dir = os.path.join(folder_name, str(label))
        os.makedirs(label_dir, exist_ok=True) #creates label subfolder if necessary
        img_path = os.path.join(label_dir, f"{i}.png")
        Image.fromarray(img).save(img_path)
        print("Image:", img_path, "is saved")

      print(f"Saved {len(total_img)} images in '{folder_name}' directory.")




"TRAINING CNN AND MLP"

def train_MLP(x_train, y_train_cat, x_test, y_test_cat):
   print("Training MLP start!")
   model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation="relu"),
    #Dense(128, activation="relu", input_shape=(784,)),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax") # for digits 0-9
    ])
   
   model.compile(optimizer = 'adam', 
                 loss = "categorical_crossentropy", 
                 metrics = ['accuracy'])
   
   history = model.fit(x_train, y_train_cat, epochs=10, batch_size=16, validation_split=0.1)
   print("Training for MLP is completed!")

   metrics = model.evaluate(x_test, y_test_cat)
   print("Results:")
   for name, value in zip(model.metrics_names, metrics):
      print(name,":",value)

   y_pred = model.predict(x_test)
   y_true = np.argmax(y_test_cat, axis = 1)
   y_pred_labels = np.argmax(y_pred, axis = 1)

   '''print("Y_TEST VALS")
   for i in range(3):
      print(np.argmax(y_test_cat[i]))
   print("Y_PRED VALS")
   for i in range(3):
      print(np.max(y_pred[i]))
   print("Y_PRED_LABELS VALS")
   for i in range(3):
      print(y_pred_labels[i])
'''
   incorrect_pred = []
   for i in range(len(y_test_cat)):
      true_val = np.argmax(y_test_cat[i])
      if true_val!=y_pred_labels[i]:
         #put the number of image in the list
         incorrect_img_index = i 
         incorrect_pred.append(incorrect_img_index)
   
   #Verifying if the image accuracy checker is working
   for i in range(3):
      #print("images", y_test_cat[i])
      print("Incorrect image:", incorrect_pred[i]+60000)
      #print("Real val:",y_test_cat[incorrect_pred[i]])
      print("Real val:",np.argmax(y_test_cat[incorrect_pred[i]]))
      #print("Predicted vals",y_pred_labels[index_incorrect])
      #print("Predictions:",y_pred[incorrect_pred[i]])
      print("Prediction perct:",np.max(y_pred[incorrect_pred[i]]))
      print("Prediction val:",np.argmax(y_pred[incorrect_pred[i]]))
      #print("Predicted vals",y_pred_labels[incorrect_pred[i]])

      
      
   

   print("MLP Classification Report:")
   print(classification_report(y_true, y_pred_labels))

   model_path = "MLP_and_CNN_training/mlp_model.h5"
   if not os.path.exists(model_path):  
        model.save(model_path)
    
   return history




def train_CNN(x_train_cnn, y_train_cat, x_test_cnn, y_test_cat):
    print("Training CNN start!")
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28, 1)), 
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, (3,3), activation = "relu"),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation='softmax')
        ])
    
    model.compile(optimizer = "adam",
                  loss = "categorical_crossentropy",
                  metrics = ['accuracy'])
    
    history = model.fit(x_train_cnn, y_train_cat, epochs=10, batch_size=16, validation_split=0.1)
    print("Training for CNN is completed!")

    metrics = model.evaluate(x_test_cnn, y_test_cat)
    print("Results:")
    for name, value in zip(model.metrics_names, metrics):
       print(name,":",value)

    y_pred = model.predict(x_test_cnn)
    y_true = np.argmax(y_test_cat, axis = 1)
    y_pred_labels = np.argmax(y_pred, axis = 1)

    print("CNN Classification Report:")
    print(classification_report(y_true, y_pred_labels))

    model_path = "MLP_and_CNN_training/cnn_model.h5"
    model.save(model_path)  

    return history


#records all the images that cause the accuracy to drop, alongside their truth/predicted labels and confidence levels
def inaccuracy_recorder(model, y_test):
   # if i in y_test doesn't != model predicition for that image ==> inaccuracy

   pass


#plots
def plot_learning_curve(history, title, save_path):
   print("MEOW")
   accuracy = history.history["accuracy"]
   val_accuracy = history.history["val_accuracy"]
   loss = history.history["loss"]
   val_loss = history.history["val_loss"]

   epochs = range(1, len(accuracy) + 1)

   plt.figure(figsize=(12, 5))

   # Accuracy 
   plt.subplot(1, 2, 1)
   plt.plot(epochs, accuracy, 'b', label="Training Accuracy")
   plt.plot(epochs, val_accuracy, 'r', label="Validation Accuracy")
   plt.title(f'{title} - Accuracy') # header 
   plt.xlabel('Epochs')
   plt.ylabel('Accuracy')
   plt.legend()

   # Loss
   plt.subplot(1, 2, 2)
   plt.plot(epochs, loss, 'b', label="Training Loss")
   plt.plot(epochs, val_loss,'r', label = "Validation Loss")
   plt.title(f'{title} - Loss') # header 
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   plt.legend()
   plt.tight_layout()
   #Save graphs
   if save_path:
        plt.savefig(save_path)
   plt.close()
   print("Graph is ready!")
   #plt.tight_layout()
   #plt.show()



"""
for f1 score too:
from sklearn.metrics import classification_report

# Predictions
y_pred_mlp = model_mlp.predict(x_test)
y_pred_cnn = model_cnn.predict(x_test_cnn)

# Convert one-hot to label
y_true = np.argmax(y_test_cat, axis=1)
y_pred_labels_mlp = np.argmax(y_pred_mlp, axis=1)
y_pred_labels_cnn = np.argmax(y_pred_cnn, axis=1)

# Print classification report
print("MLP Classification Report:\n", classification_report(y_true, y_pred_labels_mlp))
print("CNN Classification Report:\n", classification_report(y_true, y_pred_labels_cnn))

# includes precision, recall, f1-score, support
"""


#Visualize models

"""
predictions_mlp = model_mlp.predict(x_test)
pred_labels = np.argmax(predictions_mlp, axis = 1)

i = 0
plt.imshow(x_test[i], cmap='gray')
plt.title(f"True: {y_test[i]}, Predicted: {pred_labels[i]}")
plt.show()
output_file = "mlp_graph.png"
plt.savefig(output_file)
plt.close()
"""


"""
predictions_cnn = model_cnn.predict(x_test_cnn)
pred_labels = np.argmax(predictions_cnn, axis = 1)

i = 0
plt.imshow(x_test[i], cmap='gray')
plt.title(f"True: {y_test[i]}, Predicted: {pred_labels[i]}")
plt.show()
output_file = "cnn_graph.png"
plt.savefig(output_file)
plt.close()
"""

if __name__== '__main__':
    folder_name = "MLP_and_CNN_training"
    #LOAD DATA
    #By defeault, training sets have 60,000 images so 0 - 59999
    # Test sets have 10,000 images so 60000-69999
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    

    #Flatten + Normalize the images

    #x_train = x_train.reshape(-1, 28*28)
    #x_test = x_test.reshape(-1, 28*28)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32')/ 255.0

    #Images for MLP
    #x_train_mlp = x_train.reshape(-1, 28*28)
    #x_test_mlp = x_test.reshape(-1, 28*28)
    

    # Images for CNN (have channel dimension)
    x_train_cnn = x_train[..., tf.newaxis]
    #x_train_cnn = x_train.reshape(-1, 28, 28, 1)
    x_test_cnn = x_test[..., tf.newaxis]
    #x_test_cnn = x_test.reshape(-1, 28, 28, 1)
    
   
    
    # One-hot encode labels
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)


    
    #print("x_train shape:", x_train_mlp.shape)
    print("x_train shape:", x_train.shape)
    print("x_train_cnn shape:", x_train_cnn.shape)
    print("y_train_cat shape:", y_train_cat.shape)

    #print("x_test shape:", x_test_mlp.shape)
    print("x_test shape:", x_test.shape)
    print("x_test_cnn shape:", x_test_cnn.shape)    
    print("y_test_cat shape:", y_test_cat.shape)

    #for graphs
    title_mlp = "MLP"
    save_path_mlp = "mlp_graph.png"

    title_cnn = "CNN"
    save_path_cnn = "cnn_graph.png"
    #print("your mom")

    
    #download_img(folder_name)

    #train MLP
    print("MEOW2")
    #history_mlp = train_MLP(x_train_mlp, y_train_cat, x_test_mlp, y_test_cat)
    history_mlp = train_MLP(x_train, y_train_cat, x_test, y_test_cat)
    plot_learning_curve(history_mlp, title_mlp, save_path_mlp)
    tf.keras.backend.clear_session()
    print("MEOW3")
    #train CNN
    #history_cnn = train_CNN(x_train_cnn, y_train_cat, x_test_cnn, y_test_cat)  
    #plot_learning_curve(history_cnn, title_cnn, save_path_cnn)

