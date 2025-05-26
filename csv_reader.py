#Working with images:
import time
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import matplotlib.pyplot as plt

# Data handlings
#from pandas import *
import pandas as pd

#Preventing the Connection reset by peer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

#Setting up sessions with retries. Code for now, might change later
session = requests.Session()
retries = Retry(total = 3, backoff_factor = 1, status_forcelist = [500,502,503,504], raise_on_status = False )
adapter = HTTPAdapter(max_retries=retries)
session.mount('https://', adapter)



def download_img(url):

  """
  _summary_

  Args:
    url (string): url link for the image we wish to download

  Returns: 
    Image: returns the image stored at that link
  
  """


  try:
    img = session.get(url, timeout=10)
    img.raise_for_status()
    time.sleep(0.1) # prevent spamming server
    return Image.open(BytesIO(img.content))
    
  except Exception as e:
    print(f"Failed to load image: {url} — {e}")
    return None 

  """
  img = session.get(url, timeout=10)
  img.raise_for_status()
  time.sleep(0.1) # prevent spamming server
  return Image.open(BytesIO(img.content)) 

  try:
    img = session.get(url, timeout=10)
    img.raise_for_status()
    time.sleep(0.1) # prevent spamming server
    return Image.open(BytesIO(img.content))
    
  except Exception as e:
    print(f"Failed to load image: {url} — {e}")
    return None
    
  except Exception as e:
        print(f"Failed to load image: {url} — {e}")
        return None
  """


def id_extracter(url):
  """_summary_

  Args:
      url (String): url link for the image we wish to download

  Returns:
      String: returns the id associated to the original image
  """
  name = url.split("/")[-1]

  base = name.split(".")[0]

  """ if "IMG_TRANSFORMED" in url:
    base = name.split(".")[0]
  else:
    base = name.split(".")[0]
  """
  return base



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
  df['img_id'] = df['image_link'].apply(id_extracter)
  
  

  #Image Pairs:
  for img_id, group in tqdm(df.groupby('img_id'), desc="Processing image pairs"):
    origin = group[group["transformation"] == "original"]
    if len(origin) == 0:
        # Skip if no original images
        continue  

    #get the og img and download it
    origin_row = origin.iloc[0]
    img1 = download_img(origin_row["image_link"])
    if img1 is None:
        continue
    # Pair with each transformed image
    for _, row in group.iterrows():
        if row["transformation"] == "original":
            # skip original => we already have it
            continue  
        
        img2 = download_img(row["image_link"])
        if img2 is None:
          continue
        validLabel = row["ground_truth"]
        if validLabel==chosen_label:
          validLabel = 1
        else:
           validLabel = 0


        item = {'img1':img1, 'img2':img2, 'label':validLabel}
        pairs.append(item)
        #item = (img1, img2, label)
  
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
      img1 = pair_at_index['img1'] 
      img2 = pair_at_index['img2'] 
      #label = f"validity: {pair_at_index['label']}"
      label = "validity: " + str(pair_at_index['label'])
      
      #image display
      fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # 1 row, 2 columns
      axes[0].imshow(img1)
      axes[0].set_title('Original')
      axes[0].axis('off') # Hide axes

      axes[1].imshow(img2)
      axes[1].set_title('Synthesized')
      axes[1].axis('off')

      #label to figure
      fig.suptitle(label, fontsize=16)
      plt.show()

  else:
      print("Error => index out of bounds")
      return
  

  


if __name__ == '__main__':

  #You can use a different csv file. This is just a sample
  csv_file = "imagenet_experiment_results.csv"
  chosen_label = "car"
  myPairs = csv_reader(csv_file, chosen_label)
  
  while True:
        try:
            index = int(input("\nEnter index to display image pair (or -1 to exit): "))
            if index == -1:
                break
            display_img(index, myPairs)
        except ValueError:
            print("Please enter a valid integer.")


  """print(
    "List with image pairs and their validity labels: " 
  )
  
  x = 1
  for i in myPairs:
    print(f"Pair {x}: Label = {i['label']}")
    x+=1
  """






