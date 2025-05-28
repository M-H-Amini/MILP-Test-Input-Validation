#Working with images:
import time
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import matplotlib.pyplot as plt

# Data handlings
import pandas as pd
import os

#Preventing the Connection reset by peer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

#Setting up sessions with retries. Code for now, might change later
session = requests.Session()
retries = Retry(total = 3, backoff_factor = 1, status_forcelist = [500,502,503,504], raise_on_status = False )
adapter = HTTPAdapter(max_retries=retries)
session.mount('https://', adapter)


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
    for index, row in df.iterrows():
      
      img_url = row[img_url_col]
      if pd.isna(img_url): #skip url if is NaN (Not a Number)
        print("Skipping since no URL was found")
        continue
      
      try:
        resp = requests.get(img_url, stream=True)
        
        #extracting img name
        filename = os.path.join(folder_name, os.path.basename(img_url))
        
        if not filename: # If os.path.basename returns empty
          filename = os.path.join(folder_name, f"image_{index}.jpg") # Fallback filename

        with open(filename, 'wb') as out_file:
          out_file.write(resp.content)
        print(f"Download: {img_url} to {filename}")

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
        


def prefix_extracter(url):
  """_summary_

  Args:
      url (String): url link for the image we wish to download

  Returns:
      String: returns the prefix associated to a specific image
  """
   
  filename = url.split("/")[-1]
  prefix = filename.split("_")[0] + "_" + filename.split("_")[-1].split(".")[0]
  return prefix

#reader for any csv
def csv_reader(csv_path, chosen_label):

  """
  Args:
      csv_path (csv): path to the csv file with our images
      chosen_label (String): basis that will allow us to determine if the valdiity of an image is 1 or 0 

  Returns:
      list: returns list containing groups consisting of image pairs and their validity label
  """
  pairs = []
  df = pd.read_csv(csv_path)

  #new column for image ids
  df['img_prefix'] = df['image_link'].apply(prefix_extracter)
  
  

  #Image Pairs:
  for prefix, group in tqdm(df.groupby('img_prefix'), desc="Processing image pairs"):
    origin = group[group["transformation"] == "original"]
    if len(origin) == 0:
        # Skip if no original images
        continue  

    #get the og img and download it
    origin_row = origin.iloc[0]
    #img1 = download_img(origin_row["image_link"])
    img1 = origin_row["image_link"]
    if img1 is None:
        continue
    # Pair with each transformed image
    for _, row in group.iterrows():
        if row["transformation"] == "original":
            # skip original => we already have it
            continue  
        
        #img2 = download_img(row["image_link"])
        img2 = row["image_link"]
        if img2 is None:
          continue

        validLabel = row["ground_truth"]
        if validLabel==chosen_label:
          validLabel = 1
        else:
           validLabel = 0


        filename1 = os.path.basename(img1)
        filename2 = os.path.basename(img2)
        item = {'img1':filename1, 'img2':filename2, 'label':validLabel}
        pairs.append(item)
        #item = (img1, img2, label)
  

  # Debugging
  print("Sample of image pairs:")
  for i, pair in enumerate(pairs[:3]):
    print(f"Pair {i}: label={pair['label']}")
    print(f"  Original: {pair['img1']}")
    print(f"  Synthesized: {pair['img2']}")

  print(f"Loaded {len(pairs)} image pairs.")
  return pairs


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
      plt.show(block=True)

      print(f"Pair index: {index}")
      print(f"img1: {img1_path}, img2: {img2_path}")


  else:
      print("Error => index out of bounds")
      return
  

if __name__ == '__main__':

  #You can use a different csv file. This is just a sample
  csv_file_path = "imagenet_experiment_results.csv"
  chosen_label = "car"
  folder_name = "csv_images"
  img_url_col = "image_link"
  download_img(csv_file_path, folder_name, img_url_col)
  
  myPairs = csv_reader(csv_file_path, chosen_label)
  
  while True:
        try:
            index = int(input("\nEnter index to display image pair (or -1 to exit): "))
            if index == -1:
                break
            display_img(index, myPairs)
        except ValueError:
            print("Please enter a valid integer.")




