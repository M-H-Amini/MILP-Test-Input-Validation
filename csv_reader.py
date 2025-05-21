
#Working with images:
from PIL import Image
import requests
from io import BytesIO

# Data handlings
from pandas import *

#from collections import defaultdict




def download_img(url):

  """
  _summary_

  Args:
    url (string): url link for the image we wish to download

  Returns: 
    Image: returns the image stored at that link
  
  """
  img = requests.get(url)
  return Image.open(BytesIO(img.content))
 
  """
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
  img_id = name.split("_",1)[0] #separates at last underscore
  return img_id



#reader for any csv
def csv_reader(csv_path, chosen_label):

  """
  Args:
      csv_path (csv): path to the csv file with our images
      chosen_label (String): basis that will allow us to determine if the valdiity of an image is 1 or 0 

  Returns:
      list: returns list containing groups consisting of image pairs and their validity label
  """

  df = read_csv(csv_path)

  #new column for image ids
  df['img_id'] = df['image_link'].apply(id_extracter)
  pairs = []
  

  #Image Pairs:
  for img_id, group in df.groupby('img_id'):
    origin = group[group["transform"] == "original"]
    if len(origin) == 0:
        # Skip if no original images
        continue  

    #get the og img and download it
    origin_row = origin.iloc[0]
    img1 = download_img(origin_row["image_link"])

    for _, row in group.iterrows():
        if row["transform"] == "original":
            # skip original => we already have it
            continue  
        
        img2 = download_img(row["image_link"])
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




if __name__ == '__main__':

  #You can use a different csv file. This is just a sample
  csv_file = "imagenet_experiment_results.csv"
  chosen_label = "car"
  myPairs = csv_reader(csv_file, chosen_label)
  print(
    "List with image pairs and their validity labels: " 
  )
  
  x = 1
  for i in myPairs:
    print(f"Pair {x}: Label = {i['label']}")
    x+=1

  





  """

    OLD CODE:
  
  
    for i in range(0, total-1,2):
   #image pairs
    url1 = links[i]
    img1 = Image.open(BytesIO(requests.get(url1).content))
    url2 = links[i+1]
    img2 = Image.open(BytesIO(requests.get(url2).content))

    #LABEL
    validLabel = 0
    temp = valid[i] #assuming orginal image (img1) has the real label
    if (temp == chosen_label):
      #for different cases: if (temp in list with other vehicules for example)
      validLabel = 1
    else:
      validLabel = 0

    #item = {'img1': img1, 'img2':img2, 'label':validLabel}
    img = (img1, img2, validLabel)
    pairs.append(img)

  return pairs


#Assuming that the two cv files we will use are:
#cifar10_experiment_results 1
# AND
#imagenet_experiment_results 1

cifar10 = csv_reader('csv_files/cifar10_experiment_results.csv','car')
imagenet = csv_reader('csv_files/imagenet_experiment_results.csv','car')
print(cifar10)"""

"""

#read the csv files
csvFile1 = read_csv('csv_files/cifar10_experiment_results.csv')
csvFile2 = read_csv('csv_files/imagenet_experiment_results.csv')
#print(csvFile2)

#CIFAR10
links1 = csvFile1['image_link'].tolist() 
#img1 and img2?
valid1 = csvFile1['ground_truth'].tolist() 
total1 = len(links1)
cifar10 = []


#IMAGENET
links2 = csvFile2['image_link'].tolist() 
valid2 = csvFile2['ground_truth'].tolist() 
total2 = len(links2)
imagenet = []

for i in range(0,total1-1,2):

  #IMAGE PAIR
  #extracts data from link
  #assume first one is the og and second is modified
  url1 = links1[i]
  #data1 = requests.get(url1).content
  '''f = open('img'+i+'.png','wb')
  f.write(data1)
  f.close()
  '''
  img1 = Image.open(BytesIO(requests.get(url1).content))

  url2 = links1[i+1]
  #data2 = requests.get(url2).content

  img2 = Image.open(BytesIO(requests.get(url2).content))

  #LABEL
  validLabel = 0
  temp = valid1[i] #assuming orginal image (img1) has the real label
  if (temp == 'car'):
    #for different cases: if (temp in list with other vehicules for example)
    validLabel = 1
  else:
    validLabel = 0

  item = {'img1': img1, 'img2':img2, 'label':validLabel}
  #item = (img1, img2, validLabel)
  cifar10.append(item)
  print(f"Pair {i}//{i+1} - Label: {validLabel}")
  #print(item)


for i in range(0,total2-1,2):

  #IMAGE PAIR
  #extracts data from link
  url1 = links2[i]
  #data2 = requests.get(url1).content
  img1 = Image.open(BytesIO(requests.get(url1).content))

  url2 = links2[i+1]
  #data2 = requests.get(url1).content
  img2 = Image.open(BytesIO(requests.get(url2).content))


  #LABEL
  validLabel = 0
  temp = valid2[i]
  if (temp == 'car'):
    validLabel = 1
  else:
    validLabel = 0

  item = {'img1': img1, 'img2':img2, 'label':validLabel}
  #item = (img1, img2, validLabel)
  imagenet.append(item)
  #print(item)
  #print(f"Pair {i}//{i+1} - Label: {validLabel}")
  
#return for each (img1, img2, label) 
#or ('img1': img1, 'img2' : img2, 'label' : label)


"""


