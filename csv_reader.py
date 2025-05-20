
from PIL import Image
from pandas import *
import requests
from io import BytesIO

#for any csv
def csv_reader(csv_path, chosen_label):
  df = read_csv(csv_path)
  #assuming that will be the column title for the links
  links = df['image_link'].tolist()
  total = len(links)
  #assuming that will be the column title for the label
  valid = df['ground_truth'].tolist()
  pairs = []
  
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
print(cifar10)





# here is the old code, before it became a general function

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


