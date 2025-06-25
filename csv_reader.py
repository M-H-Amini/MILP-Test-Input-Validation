#Working with images:
import time
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Data handlings
import pandas as pd
import os

#importing functions from other files
from computeMetrics import *


#For working with tuples
from collections import Counter

#Preventing the Connection reset by peer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

#Setting up sessions with retries. Code for now, might change later
session = requests.Session()
retries = Retry(total = 3, backoff_factor = 1, status_forcelist = [500,502,503,504], raise_on_status = False )
adapter = HTTPAdapter(max_retries=retries)
session.mount('https://', adapter)


##  TODO:  Use tqdm to show download progress in a single line instead of printing many lines
def download_img(csv_file_path, folder_name="csv_images", img_url_col="image_link"):
  
  """
  _summary_

  Args:
    csv_file_path (str): the path to the csv_file
    folder_name (str): the name of the folder where the downloaded images will be stored
    img_url_col (str): the name of column where the image urls are stored

  Returns: 
    None: the newly made folder gets populated with images
  
  """

  if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print("Folder is created: " + folder_name)
  else:
    print("Folder "+ folder_name + " already exists")
    return

  try:
    df = pd.read_csv(csv_file_path)
    
    if img_url_col not in df.columns:
      print("Error: " + img_url_col + " is not a column in the csv_file")
      return
    
    #download img from each row
    for index, row in tqdm(df.iterrows(), total=len(df), desc = "Downloading Images: "):
      
      img_url = row[img_url_col]
      if pd.isna(img_url): #skip url if is NaN (Not a Number)
        #("Skipping since no URL was found")
        continue
      
      try:
        #resp = requests.get(img_url, stream=True)
        resp = session.get(img_url, stream=True)
        #extracting img name
        filename = os.path.join(folder_name, os.path.basename(img_url))
        
        if not filename: # If os.path.basename returns empty
          filename = os.path.join(folder_name, f"image_{index}.jpg") # Fallback filename

        with open(filename, 'wb') as out_file:
          out_file.write(resp.content)
        #print(f"Download: {img_url} to {filename}")
        #tqdm.write(f"Download: {img_url} to {filename}")

      except requests.exceptions.RequestException as e:
        print(f"Error downloading {img_url}: {e}")
      except Exception as e:
        print(f"An unexpected error occurred for {img_url}: {e}")


  except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_file_path}")
  except pd.errors.EmptyDataError:
    print(f"Error: CSV file is empty at {csv_file_path}")
  except Exception as e:
    print(f"An error occurred while reading the CSV: {e}")
        

##  TODO:  Instead of _summary_, please provide a more descriptive docstring saying what is an id and how it is extracted
def id_extracter(url):

  """
  Here, an 'image_id' refers to the id from the original original. We will need to exract it
  from the original image's url to pair the original images only with their synthesized counterparts.

  For example, given this original image url: https://pilotexp.s3.ca-central-1.amazonaws.com/IMG_ORIGINAL/defocus_blur/n03095699_3680.JPEG
  we would only take the basename of the link "n03095699_3680.JPEG", and use the .split function for "_".
  This will will give the list ["n03095699", "3680.JPEG"]. 

  In order to get the image_id, we will concatenate the items at index 0 and 1.
  The reason why we don't just use the item at the last index is because synthesized images, unlike original ones, 
  have more than two items in their lists after the .split, so accessing the last item will result in an incorrect id. 
  ex: list after .split on synthesized image's basename = ['n03127747', '4845.JPEG.gaussian', 'noise', '816.png']
    => the correct image_id = "n03127747_4845", but if we concatenate only the first and last items, we would get image_id = "n03127747_816"
  
  Since we desire to get the id "n03095699_3680", we'll concatenate those first two items,
  and get rid of the of the .JPEG" from the second one by using the .split function again,
  only this time for the ".". This will generate the list ["3680", "JPEG"] but we'll only use the first value.

  Thus:
  filename = os.path.basename(url) # extracting the basename
  image_id = filename.split("_")[0] + "_" + filename.split("_")[1].split(".")[0] # AKA "n03095699" + "_" + "3680"
    
    => the image_id would be "n03095699_3680".
    
  """
  
  filename = os.path.basename(url)
  img_id = filename.split("_")[0] + "_" + filename.split("_")[1].split(".")[0] # creates string representing the id
  return img_id


#reader for any csv
def csv_reader(csv_path):

  """
  Args:
      csv_path (csv): path to the csv file with our images
      chosen_label (String): basis that will allow us to determine if the valdiity of an image is 1 or 0 

  Returns:
      list: returns list containing groups consisting of image pairs and their validity label
  """
  pairs = []
  pair_transforms = []
  df = pd.read_csv(csv_path)

  #new column for image ids
  df['img_id'] = df['image_link'].apply(id_extracter)
  

  #Image Pairs:
  for img_id, group in tqdm(df.groupby('img_id'), desc="Processing image pairs"):
    origin = group[group["transformation"] == "original"]
    if len(origin) == 0:
        # Skip if no original images
        continue  

    #get the og img and download it
    origin_row = origin.iloc[0]
    img1 = origin_row["image_link"]
    transformation1 = origin_row["transformation"]

    if img1 is None:
        continue
    # Pair with each transformed image
    for _, row in group.iterrows():
        if row["transformation"] != "original":
            # skip original => we already have it
            img2 = row["image_link"]
            transformation2 = row["transformation"]
            
        else:
          continue  

        if img2 is None:
          continue

        validLabel1 = row["ground_truth"]
        validLabel2 = row["human_label"]

        validLabel = 0
        if validLabel1==validLabel2:
          validLabel = 1
        else:
           validLabel = 0

        filename1 = os.path.basename(img1)
        filename2 = os.path.basename(img2)
        item = {'img1':filename1, 'img2':filename2, 'label':validLabel}
        check = {'img1':transformation1, 'img2':transformation2}
        pairs.append(item)
        pair_transforms.append(check)
        #item = (img1, img2, label)
  

  # Debugging
  """print("Sample of image pairs:")
  for i, pair in enumerate(pairs[:3]):
    print(f"Pair {i}: label={pair['label']}")
    print(f"  Original: {pair['img1']}")
    print(f"  Synthesized: {pair['img2']}")
"""
  print(f"Loaded {len(pairs)} image pairs.")
  return pairs,  pair_transforms #change later => for debugging



#displays image pair at specific index, with label
def display_img(index, myPairs):

  '''
  _summary_

  Args: 
    index (int): index of a specific pair in our pairs list
    myPairs (list): list of all image pairs found in the csv file
  
  Returns: 
    none: only a plot of the two images and a label defining the validity
  '''
  biggest = len(myPairs)

  if 0 <= index < biggest:
      
      pair_at_index = myPairs[index] 
      img1_path = os.path.join("csv_images", pair_at_index['img1'])
      img2_path = os.path.join("csv_images", pair_at_index['img2'])
      #label = f"validity: {pair_at_index['label']}"
      if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Image file(s) not found:\n  - {img1_path}\n  - {img2_path}")
        return
      
      #access downloaded images via path:
      try:
        
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
      except Exception as e:
        print(f"Error opening image(s): {e}")
        return
      

      # Plotting
      plt.ioff()
      fig, axes = plt.subplots(1, 2, figsize=(10, 5))
      axes[0].imshow(img1)
      axes[0].set_title('Original')
      axes[0].axis('off')

      axes[1].imshow(img2)
      axes[1].set_title('Synthesized')
      axes[1].axis('off')

      fig.suptitle(f"Validity: {pair_at_index['label']}", fontsize=16)
      plt.tight_layout()
      #plt.show(block=True)
      output_file = f"display_pair_{index}.png"
      plt.savefig(output_file)
      plt.close()
      print(f"Pair index: {index}")
      print(f"img1: {img1_path}, img2: {img2_path}")


  else:
      print("Error => index out of bounds")
      return
  

def dataset_split(percent, myPairs):

    """
      _summary_
        Args:
            percent (float): % of the original dataset (in decimal) that we will allocate to the test set
            myPairs (list): image pairs generated from the image dataset

        Returns: 
            List: returns the training and testing dataset
    """
    ds_train, ds_test = train_test_split(
      myPairs,
      test_size = percent, 
      random_state= 42
    )
    print("Training data set: ", len( ds_train)," pairs")
    print("Testing data set: ", len( ds_test), " pairs")

    return ds_train, ds_test



def unique_pair_dataset(dataset):
  ds_unique = []
  print("inside the unique pair function")
  myTuples = [dict_to_tuple(pair) for pair in dataset]  
  pair_counts = Counter(myTuples)

  for pair_tuple in pair_counts:
    print("inside the pair_tuple for-loop")
    ds_unique.append(dict(pair_tuple))
    #print(dict(pair_tuple))
    
  return ds_unique

 
def compute_metrics_on_dataset(dataset, ds_folder='csv_images'): #is this how you train it?
  
  """
  _summary_

  Args:
    training_dataset (list): dataset with image pairs made for training

  Returns: 
    Matrix: returns the matrix containing the computed metrics for all the image pairs
  """
  
  X = []
  y = []

  for pair in dataset:

    imgA_path = os.path.join(ds_folder, pair['img1'])
    imgB_path = os.path.join(ds_folder, pair['img2'])

    """imgA_path = pair['img1'] 
    imgB_path = pair['img2']"""

    imgA = Image.open(imgA_path)
    imgB = Image.open(imgB_path)
    numpy_arrayA = np.array(imgA)
    numpy_arrayB = np.array(imgB)
    if numpy_arrayA.shape != numpy_arrayB.shape:
      print(f"Skipping mismatched pair: {pair['img1']} and {pair['img2']}")
      continue
    pair_metrics = computeMetrics(numpy_arrayA, numpy_arrayB)
    X.append(pair_metrics)
    y.append(pair['label'])
  
  X = np.array(X)
  y = np.array(y)

  return X,y



def dict_to_tuple(pair):
  return tuple(sorted(pair.items()))

if __name__ == '__main__':

  #You can use a different csv file. This is just a sample
  csv_file_path = "imagenet_experiment_results.csv" 
  folder_name = "csv_images"
  img_url_col = "image_link"
  percent = 0.2
  
  download_img(csv_file_path, folder_name, img_url_col)
  
  myPairs, myChecks = csv_reader(csv_file_path)
  
  train_dataset, test_dataset = dataset_split(percent, myPairs)
  """computed_metrics = compute_metrics_on_dataset(train_dataset)
  print("Computed metrics for all pairs: ")
  print(computed_metrics)"""

  '''

  totalValid = 0
  totalInvalid = 0
  for i in myPairs:
    if i['label'] == 1:
      totalValid+=1
    else:
      totalInvalid+=1
  print("Total valid pairs: ", totalValid)
  print("Total invalid pairs", totalInvalid)

  
  # check for duplicates
  totalDuplicates = 0

  #converts pairs, that were dictionarries, to tuples.  
  myTuples = [dict_to_tuple(pair) for pair in myPairs]

  pair_counts = Counter(myTuples)
  for pair_tuple, count in pair_counts.items():
    if count > 1:
      totalDuplicates += (count-1)
  
  print("Total number of duplicates: ", totalDuplicates)'''

  
  # check for duplicates
  totalDuplicates = 0

  #converts pairs, that were dictionarries, to tuples.  
  myTuples = [dict_to_tuple(pair) for pair in myPairs]
  uniquePairs = []
  totalUnique = 0
  pair_counts = Counter(myTuples)

  for pair_tuple, count in pair_counts.items():
    if count > 1:
      totalDuplicates += (count-1)
      totalUnique += 1
    elif count == 1:
      totalUnique += 1

  for pair_tuple in pair_counts:
      uniquePairs.append(dict(pair_tuple))

  totalUnique = len(uniquePairs)
  
  totalValid = 0
  totalInvalid = 0
  for i in myPairs:
    if i['label'] == 1:
      totalValid+=1
    else:
      totalInvalid+=1

  totalValidUnique = 0
  totalInvalidUnique = 0
  for i in uniquePairs:
    if i['label'] == 1:
      totalValidUnique+=1
    else:
      totalInvalidUnique+=1
  
  print("Total valid pairs: ", totalValid)
  print("Total invalid pairs", totalInvalid)
  print("Total number of duplicates: ", totalDuplicates)
  print("Total number of unique pairs: ", totalUnique)
  print("Total number of valid pairs (no duplicates):",totalValidUnique )
  print("Total invalid pairs (no duplicates): ", totalInvalidUnique)


  while True:
        try:
            index = int(input("\nEnter index to display image pair (type -1 to exit): "))
            if index == -1:
                break
            display_img(index, myPairs)
            print(myChecks[index])
        except ValueError:
            print("Please enter a valid integer.")
