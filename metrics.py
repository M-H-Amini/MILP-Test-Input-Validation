# Core image processing + plotting
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Data handling
import pandas as pd

# Progress bar
from tqdm import tqdm


# Deep learning
import tensorflow.keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
#from tensorflow.keras.losses import mean_squared_error as mse
#commented out since we already manually calculate the mse


# ML utilities
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.model_selection import train_test_split

# Stat methods
from scipy.special import rel_entr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import wasserstein_distance
from scipy.stats import pearsonr

# Image similarity and analysis
from skimage.metrics import structural_similarity 
from skimage import data, img_as_float
from skimage import io, color, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops


# PyTorch segmentation model
from torchvision.io.image import read_image
from torchvision.io.image import ImageReadMode
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
import torch



#####################################################
"""
NOTE:

"""
#####################################################



# Global Setup
csv_file = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\cifar dataset\\cifar_dataset_modified.csv"
df = pd.read_csv(csv_file)


def computeMetrics(img_A, img_B):
  """_summary_

  Args:
      img_A (np.array): The first image
      img_B (np.array): The second image

  Returns:
      np.array: The metrics of the two images  
  """
  input_shape = (32,32)

  #resizing our images:
  img_A_new = cv2.resize(img_A,input_shape) #assuming (32,32) is the resolution we want
  img_B_new = cv2.resize(img_B,input_shape)

#CS
  model = VGG16(include_top=False, input_shape =input_shape)
  #Only extracts mid to low level features (no deep ones)
  features = model(inputs=model.input, outputs=model.get_layer("block2_conv2").outputs)
  
  #preprocesses img_A and img_B to be used by VGG16
  #via converting RBG to BGR, scaling and zero-centering with respected to Imagenet datatset
  pre_A = preprocess_input(img_A_new)
  pre_B = preprocess_input(img_B_new)

  
  batch = np.stack([pre_A,pre_B])
  f_features = features.predict(batch)
  f_A, f_B = f_features[0].flatten(), f_features[1].flatten()
  
  """
  f_A = features.predict(pre_A.reshape(1, 32, 32, 3)).flatten().reshape(1, -1)
  f_B = features.predict(pre_B.reshape(1, 32, 32, 3)).flatten().reshape(1, -1) #convert to ont-to-many
  
  """
  
  #orthogonal (no similarities)
  cs_result = cosine_similarity([f_A],[f_B])[0][0] 

#CPL
  num = (f_A - f_B)**2 
  cpl_result = np.mean(num)


#hist_cmp
  
  hist_corr = cv2.compareHist(hist_A.reshape(-1,1), hist_B.reshape(-1,1), cv2.HISTCMP_CORREL)
  hist_inter = cv2.compareHist(hist_A.reshape(-1,1), hist_B.reshape(-1,1), cv2.HISTCMP_INTERSECT)

#kl
#calculates how the much the hisotgram of A differs from histogram of B
  #converts to grayscale
  g_A = cv2.cvtColor(img_A_new, cv2.COLOR_BGR2GRAY)
  g_B = cv2.cvtColor(img_B_new, cv2.COLOR_BGR2GRAY)
  #calculates intensity (in grayscale) of pixel to compare, puts it into a 1D array
  hist_A = cv2.calcHist([g_A], [0], None, [256], [0,256])[:, 0] + 1e-10 #last part added to avoid division by 0 issues.
  hist_B = cv2.calcHist([g_B], [0], None, [256], [0,256])[:, 0] + 1e-10
 
 #normalize the histograms + avoid division by 0 issues
  hist_A = np.clip(hist_A/np.sum(hist_A), 1e-10, None)
  hist_B = np.clip(hist_B/np.sum(hist_B), 1e-10, None)

  kl_result = np.sum(rel_entr(hist_A, hist_B))


#mse
  mse_result = my_mse(img_A_new, img_B_new)

#psnr
  psnr_result = cv2.PSNR(img_A_new, img_B_new)

#ssim
  img_A_float = img_as_float(img_A_new)
  img_B_float = img_as_float(img_B_new)
  ssim_result = structural_similarity (img_A_float, img_B_float, data_range=img_B_float.max()-img_B_float.min())

#sss

  weights = FCN_ResNet50_Weights.DEFAULT
  model = fcn_resnet50(weights=weights)
  model.eval()

  #normalizes, resizes and converts images
  tensor_A = weights.transforms()(read_image(img_A, mode=ImageReadMode.RGB)).unsqueeze(0)
  tensor_B = weights.transforms()(read_image(img_B, mode=ImageReadMode.RGB)).unsqueeze(0)

  #out_A = model(tensor_A)["out"].detach()
  #out_B = model(tensor_B)["out"].detach()
  
  #no gradient tracking to save memory
  with torch.no_grad():
    out_A = model(tensor_A)["out"]
    out_B = model(tensor_B)["out"]

  mask_A = out_A.argmax(1).squeeze().numpy()
  mask_B = out_B.argmax(1).squeeze().numpy()
  sss_result = np.mean((mask_A - mask_B) ** 2)

#tsi
  angles = [0] #can be more
  distances = [5] #can be more

  #assuming that distances = [5] (5px apart) and angles = [0] (horizontal to each other)
  glcm_A = compute_glcm_features(img_A_new, distances=distances, angles=angles) 
  glcm_B = compute_glcm_features(img_B_new, distances=distances, angles=angles) 

  glcm_contrast = abs(graycoprops(glcm_A, 'contrast')[0, 0] - graycoprops(glcm_B, 'contrast')[0, 0])
  glcm_dissim = abs(graycoprops(glcm_A, 'dissimilarity')[0, 0] - graycoprops(glcm_B, 'dissimilarity')[0, 0])

  tsi_result = (glcm_contrast + glcm_dissim)/2

  """
  ubyte_A = img_as_ubyte(color.rgb2gray(img_A_new))
  ubyte_B = img_as_ubyte(color.rgb2gray(img_B_new))
  glcm_A = graycomatrix(ubyte_A, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
  glcm_B = graycomatrix(ubyte_B, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
  glcm_contrast = abs(graycoprops(glcm_A, 'contrast')[0, 0] - graycoprops(glcm_B, 'contrast')[0, 0])
  glcm_dissim = abs(graycoprops(glcm_A, 'dissimilarity')[0, 0] - graycoprops(glcm_B, 'dissimilarity')[0, 0])

  """
 
#wd

  wd_result = wasserstein_distance(g_A.flatten(), g_B.flatten())

#no vif => only gotten from a csv file.

  metrics = np.array([cpl_result, cs_result, kl_result, mse_result, hist_corr, hist_inter,psnr_result,ssim_result, sss_result, tsi_result, wd_result])
  return metrics




if __name__ == "__main__":


  """ _description_
  
  Sets your CV file in order to extract your two images.
  
  Then, assigns img_A and img_B to an image respectively, 
  then computes the associated metrics using the computeMetrics function.
  Although you would need to extract these two images from a csv file,
  for the sake of demonstrating how the function computeMetrics works,
  we have provided a ready-made sample-pair for you to use.

  You would need to use these lines for the images if you were to use it:
    img_A = cv2.imread(" path of image ")
    img_B = cv2.imread(" path of image ") 
     
  """

  csv_file = "imagenet_experiment_results.csv"
  df = pd.read_csv(csv_file)

  
  image_pathA = "sampleImages/n02009912_5558.jpg" 
  img_A = Image.open(image_pathA)
  #img_A.show()

  image_pathB = "sampleImages/n02009912_14750.JPEG.gaussian_noise_932.png"  # Replace with the actual path to your image
  img_B = Image.open(image_pathB)
  #img_B.show()
  
  metrics = computeMetrics(img_A, img_B)
  #[cpl_result, cs_result, kl_result, mse_result, hist_corr, hist_inter,psnr_result,ssim_result, sss_result, tsi_result, wd_result]
  print("The Computed Metrics for Image A and Image B: "+
        "CPL: "+str(metrics[0])
        +", CS: " + str(metrics[1])
        +", KL: " + str(metrics[2])
        +", MSE: " + str(metrics[3])
        +", HIST_CORR: " + str(metrics[4])
        +", HIST_INTER: " + str(metrics[5])
        +", PSNR: " + str(metrics[6])
        +", SSIM: " + str(metrics[7])
        +", SSS: " + str(metrics[8])
        +", TSI: " + str(metrics[9])
        +", WD: " + str(metrics[10]))



"""


# The code for majority of these metrics can be found in ICSE2025Industry.
# Classifier Perceptual Loss using VGG16

def cpl(csv_file,df):
    dataset_length = len(df)
    img_org, img_gen = [], []
    img_shape = (32, 32) 
    for i in tqdm(range(dataset_length)):
      img_org_ = cv2.resize(cv2.imread(df["original"][i]), img_shape).astype(float)
      img_org.append(img_org_)
      img_gen_ = cv2.resize(cv2.imread(df["gen"][i]), img_shape).astype(float)
      img_gen.append(img_gen_)
    
    img_org = np.array(img_org)
    img_gen = np.array(img_gen)
    print(img_org.shape, img_gen.shape)
    
    # Model:
    model = VGG16(weights='imagenet', include_top=False, input_shape=(*img_shape[::-1], 3))

    X_org = preprocess_input(img_org)
    X_gen = preprocess_input(img_gen)
    X_org = model.predict(X_org)
    X_gen = model.predict(X_gen)

    X_org = X_org.reshape(len(X_org), -1)
    X_gen = X_gen.reshape(len(X_gen), -1)

    losses = np.array(mse(X_org, X_gen))

    df["CPL"] = losses
      #got rid of the print statement for this and the others.
    df.to_csv(csv_file)


# Cosine Similarity
def cs(csv_file,df):
  dataset_length = len(df)
  img_org, img_gen = [], []
  img_shape = (32, 32) 
  for i in tqdm(range(dataset_length)):
    img_org_ = cv2.resize(cv2.imread(df["original"][i]), img_shape).astype(float)
    img_org.append(img_org_)
    img_gen_ = cv2.resize(cv2.imread(df["gen"][i]), img_shape).astype(float)
    img_gen.append(img_gen_)
  
  img_org = np.array(img_org)
  img_gen = np.array(img_gen)
  print(img_org.shape, img_gen.shape)
  
  #  Model:
  model = VGG16(weights='imagenet', include_top=False, input_shape=(*img_shape[::-1], 3))
  
  X_org = preprocess_input(img_org)
  X_gen = preprocess_input(img_gen)
  X_org = model.predict(X_org)
  X_gen = model.predict(X_gen)

  X_org = X_org.reshape(len(X_org), -1)
  X_gen = X_gen.reshape(len(X_gen), -1)

  cs = cosine_similarity(X_org, X_gen)
  cs = [cs[i,i] for i in range(len(cs))]

  df["CS"] = cs
  df.to_csv(csv_file)

# Histogram comparison using correlation and intersection
def hist_cmp(csv_file,df):
  
  dataset_length = df.shape[0]
  hist_org = []
  hist_gen = []
  correlation = []
  intersection = []
  for i in tqdm(range(dataset_length)):
    img_org = cv2.imread(df["original"][i], cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([img_org], [0], None, [256], [0, 256])
    hist_org.append(hist)
    img_gen = cv2.imread(df["gen"][i], cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([img_gen], [0], None, [256], [0, 256])
    hist_gen.append(hist)
    correlation.append(cv2.compareHist(hist_gen[i], hist_org[i], cv2.HISTCMP_CORREL))
    intersection.append(cv2.compareHist(hist_gen[i], hist_org[i], cv2.HISTCMP_INTERSECT))
  
    
  df["Hist_cor"] = correlation
  df["Hist_int"] = intersection
  df.to_csv(csv_file)
  
# Plot func from ICSE2025Industry => used in kl(csv_file,df)
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


# Kullback-Leibler divergence between image histograms 
def kl(csv_file,df):
  dataset_length = df.shape[0]
  kl = []
  for i in tqdm(range(dataset_length)):
    img_org = rgb2gray(plt.imread(df["original"][i]))
    out_org = plt.hist(x=img_org.ravel(), bins=256, range=[0, 256])
    dist_org = out_org[0]/(img_org.shape[0]*img_org.shape[1])
    img_gen = rgb2gray(plt.imread(df["gen"][i]))
    out_gen = plt.hist(x=img_gen.ravel(), bins=256, range=[0, 256])
    dist_gen = out_gen[0]/(img_gen.shape[0]*img_gen.shape[1])
    kl.append(sum(rel_entr(dist_org, dist_gen)))
  df["KL"] = kl
  df.to_csv(csv_file)

# Function from ICSE2025Industry => used in mse(csv_file,df)
def my_mse(img1, img2):
    h, w, c = img1.shape
    err = np.sum(cv2.subtract(img1, img2)**2)
    return err/float(w*h*c)


def mse(csv_file,df):

  dataset_length = df.shape[0]
  mse = []
  for i in tqdm(range(dataset_length)):
    img_org = cv2.resize(cv2.imread(df["original"][i]), img_gen) 
    img_gen = cv2.resize(cv2.imread(df["gen"][i]), img_gen) 
    mse.append(my_mse(img_org, img_gen))

  df["MSE"] = mse
  df.to_csv(csv_file, index = False)

#Peak signal-to-noise ratio
def psnr(csv_file,df):
  dataset_length = df.shape[0]
  psnr = []
  for i in tqdm(range(dataset_length)):
    img_org = cv2.imread(df["original"][i])
    img_gen = cv2.imread(df["gen"][i])
    psnr.append(cv2.PSNR(img_org, img_gen))
  df["PSNR"] = psnr
  df.to_csv(csv_file)

def ssim(csv_file,df):
  dataset_length = df.shape[0]
  ssim = []
  for i in tqdm(range(dataset_length)):
    img_org = img_as_float(cv2.imread(df["original"][i], cv2.IMREAD_GRAYSCALE))
    img_gen = img_as_float(cv2.imread(df["gen"][i], cv2.IMREAD_GRAYSCALE))
    ssim.append(structural_similarity (img_org, img_gen, data_range=img_gen.max()-img_gen.min()))
  df["SSIM"] = ssim
  df.to_csv(csv_file)

def sss(csv_file,df):
  #1: Mode initializes with best available weights
  weights = FCN_ResNet50_Weights.DEFAULT
  model = fcn_resnet50(weights=weights)
  model.eval()
  #2: Inference transforms are initialized
  preprocess = weights.transforms()
  dataset_length = df.shape[0]
  sss = []

  for i in tqdm(range(dataset_length)):
    img_org = read_image(df["original"][i]) 
    img_gen = read_image(df["gen"][i])

    #3: Inference preprocessing transforms are applied
    batch_org = preprocess(img_org).unsqueeze(0)
    batch_gen = preprocess(img_gen).unsqueeze(0)

    #4: Use the model and visualize the prediction
    prediction_org = model(batch_org)["out"]
    normalized_masks_org = prediction_org.softmax(dim=1)[0, 0]
    prediction_gen = model(batch_gen)["out"]
    normalized_masks_gen = prediction_gen.softmax(dim=1)[0, 0]
    gen = normalized_masks_gen.detach().numpy() 
    org = normalized_masks_org.detach().numpy() 
    loss = np.square(np.subtract(org, gen)).mean() 
    sss.append(loss)
  
  df["SSS"] = sss
  df.to_csv(csv_file)

# Function from ICSE2025Industry => used in tsi(csv_file,df)
def compute_glcm_features(image, distances, angles):
 # Compute GLCM
    gray_image = img_as_ubyte(color.rgb2gray(image))
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    # GLCM properties computed
    contrast = graycoprops(glcm, prop='contrast')
    dissimilarity = graycoprops(glcm, prop='dissimilarity')
    homogeneity = graycoprops(glcm, prop='homogeneity')
    energy = graycoprops(glcm, prop='energy')
    correlation = graycoprops(glcm, prop='correlation')
    metrics = (dissimilarity,)
    return metrics

# Texture similarity using gray level co-occurrence matrix
def tsi(csv_file,df):
  img_shape = (512, 320) 
  dataset_length = len(df)
  img_org, img_gen = [], []
  similarity = []

  for i in tqdm(range(dataset_length)):
    img_org_ = cv2.resize(cv2.imread(df["original"][i]), img_shape)
    img_org.append(img_org_)
    img_gen_ = cv2.resize(cv2.imread(df["gen"][i]), img_shape)
    img_gen.append(img_gen_)
    f1 = compute_glcm_features(img_org_, distances=[1], angles=[0])
    f2 = compute_glcm_features(img_gen_, distances=[1], angles=[0])
    similarity.append(np.linalg.norm(np.array(f1) - np.array(f2)))
    
  img_org = np.array(img_org)
  img_gen = np.array(img_gen)
  print(img_org.shape, img_gen.shape)
  
  df["TSI"] = similarity
  df.to_csv(csv_file, index=False)


# Wasserstein distance between grayscale images
def wd(csv_file,df):
  dataset_length = df.shape[0]
  wd = []
  for i in tqdm(range(dataset_length)):
    img_org = cv2.imread(df["original"][i], cv2.IMREAD_GRAYSCALE).flatten()
    img_gen = cv2.imread(df["gen"][i], cv2.IMREAD_GRAYSCALE).flatten()
    wd.append(wasserstein_distance(img_org, img_gen))
  df["WD"] = wd
  df.to_csv(csv_file)
  

# Variance Inflation Factor for feature multicollinearity
def vif(csv_file):
  df = pd.read_csv(csv_file)
  num_df = df.select_dtypes(include=[float, int])
 
  vif = pd.DataFrame()
  vif["feature"] = num_df.columns
  vif["VIF"] = [variance_inflation_factor(num_df.values, i) for i in range(num_df.shape[1])]

  vif.to_csv("vif_results.vsf", index=False)
  #should we return vif?





    


  







