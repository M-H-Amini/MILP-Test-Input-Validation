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



# Collect all the incorrectly predicted images
   incorrect_pred = []
   for i in range(len(y_test_cat)):
      true_val = y_true[i]
      if true_val!=y_pred_labels[i]:
         #put the number of image in the list
         incorrect_img_index = i 
         incorrect_pred.append(incorrect_img_index)
   
   # Prepare dictionary will all needed values for the html file
   inaccurate_imgs_for_html = []
   for i in range(len(incorrect_pred)):
      real_label = y_true[incorrect_pred[i]]
      img_url = "MLP_and_CNN_training/"+str(real_label)+"/"+str(incorrect_pred[i]+60000)+".png"
      #img_url = "MLP_and_CNN_training/MLP_and_CNN_training/"+str(real_label)+"/"+str(incorrect_pred[i]+60000)+".png"
      #img_url = "/home/alina/MLP_and_CNN_training/MLP_and_CNN_training/"+str(real_label)+"/"+str(incorrect_pred[i]+60000)+".png"
      pred_label = y_pred_labels[incorrect_pred[i]]
      pred_perct = np.max(y_pred[incorrect_pred[i]])
      img_info = {"img_url":img_url, "real_label": real_label, "predicted_label": pred_label, "confidence":pred_perct}
      inaccurate_imgs_for_html.append(img_info)
   
      
   #Verifying if the image accuracy checker is working
      '''#print("images", y_test_cat[i])
      print("Incorrect image:", incorrect_pred[i]+60000)
      #print("Real val:",y_test_cat[incorrect_pred[i]])
      print("Real val:",np.argmax(y_test_cat[incorrect_pred[i]]))
      #print("Predicted vals",y_pred_labels[index_incorrect])
      #print("Predictions:",y_pred[incorrect_pred[i]])
      print("Prediction perct:",np.max(y_pred[incorrect_pred[i]]))
      print("Prediction val:",np.argmax(y_pred[incorrect_pred[i]]))
      #print("Predicted vals",y_pred_labels[incorrect_pred[i]])
      '''

   
   #print("MLP Classification Report:")
   #print(classification_report(y_true, y_pred_labels))

   model_path = "MLP_and_CNN_training/mlp_model.h5"
   if not os.path.exists(model_path):  
        model.save(model_path)
    
   return history, inaccurate_imgs_for_html



'''def inaccurate_img_urls(incorrect_pred, y_test_cat, y_pred):
   # incorrect_pred - contains all the incorrect images, but you'll need to add 60000
   #Will access the images from the folder
   #makes table with image, real label, predicdted label and confidence level
   total_inaccurate_img_url = []
   for i in incorrect_pred:
      img_url = str(incorrect_pred[i]+60000)+".png"
      total_inaccurate_img_url.append(img_url)
   return total_inaccurate_img_url
'''
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

    #print("CNN Classification Report:")
    #print(classification_report(y_true, y_pred_labels))

    # Collect all the incorrectly predicted images
    incorrect_pred = []
    for i in range(len(y_test_cat)):
      #true_val = np.argmax(y_test_cat[i])
      true_val = y_true[i]
      if true_val!=y_pred_labels[i]:
         #put the number of image in the list
         incorrect_img_index = i 
         incorrect_pred.append(incorrect_img_index)
   
   # Prepare dictionary will all needed values for the html file
    inaccurate_imgs_for_html = []
    for i in range(len(incorrect_pred)):
      real_label = y_true[incorrect_pred[i]]
      #MLP_and_CNN_training/MLP_and_CNN_training/9/4.pngs
      #img_url = "MLP_and_CNN_training/MLP_and_CNN_training/"+str(real_label)+"/"+str(incorrect_pred[i]+60000)+".png"
      img_url = "MLP_and_CNN_training/"+str(real_label)+"/"+str(incorrect_pred[i]+60000)+".png"
      #img_url = "/home/alina/MLP_and_CNN_training/MLP_and_CNN_training/"+str(real_label)+"/"+str(incorrect_pred[i]+60000)+".png"
      pred_label = y_pred_labels[incorrect_pred[i]]
      pred_perct = np.max(y_pred[incorrect_pred[i]])
      img_info = {"img_url":img_url, "real_label": real_label, "predicted_label": pred_label, "confidence":pred_perct}
      inaccurate_imgs_for_html.append(img_info)

    model_path = "MLP_and_CNN_training/cnn_model.h5"
    model.save(model_path)  
   #MLP_and_CNN_training/MLP_and_CNN_training/9/4.png
    return history, inaccurate_imgs_for_html


#records all the images that cause the accuracy to drop, alongside their truth/predicted labels and confidence levels
def html_builder(inaccurate_imgs_for_html, model_name):
   # will access each dictionary and populate the table
   # 
   #   
   html_content ="""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title> Image table </title>
        <style>
            table {
               border-collapse: collapse;
            }
            td {
               border: 1px solid black;
               padding: 10px;
               text-align: center;
            }
            img {
               width: 50px;
               height: auto;
            }
        </style>
    </head>
    <body>
      <h2>Inaccurately Predicted Images</h2>
      <table>
         <tr>
            <th> Image </th>
            <th> True Label </th>
            <th> Predicted Label </th>
            <th> Confidence </th>
         </tr>
    """
   for i in inaccurate_imgs_for_html:
      html_content += f"""
      <tr>
         <td><img src = "{i['img_url']}" alt = "inaccurate img" width = "50" loading="lazy" ></td>
         <td>{i['real_label']}</td>
         <td>{i['predicted_label']}</td>
         <td>{i['confidence']:.2f}</td>
      </tr>
      """
   html_content += """
   </table>
   </body>
   </html>
   """
   html_name = model_name+"_inaccurate_img_table.html"
   with open(html_name, "w", encoding = "utf-8") as f:
      f.write(html_content)

   print("HTML table created: ", html_name)
   


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
    #print("MEOW2")
    #history_mlp = train_MLP(x_train_mlp, y_train_cat, x_test_mlp, y_test_cat)
    history_mlp, inaccurate_imgs_mlp = train_MLP(x_train, y_train_cat, x_test, y_test_cat)
    #plot_learning_curve(history_mlp, title_mlp, save_path_mlp)
    print(inaccurate_imgs_mlp[0])
    tf.keras.backend.clear_session()
    
    #print("MEOW3")
    #train CNN
    history_cnn, inaccurate_imgs_cnn = train_CNN(x_train_cnn, y_train_cat, x_test_cnn, y_test_cat)  
    #plot_learning_curve(history_cnn, title_cnn, save_path_cnn)


   #build our html file

   #For MLP
    html_builder(inaccurate_imgs_mlp, title_mlp)

   #For CNN
    html_builder(inaccurate_imgs_cnn, title_cnn)

    
    
