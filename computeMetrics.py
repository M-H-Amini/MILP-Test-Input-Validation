# Core image processing + plotting
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Data handling
import pandas as pd

# Progress bar
from tqdm import tqdm

# Handle images
from PIL import Image

# Deep learning
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

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


# Global model definitions (load only once)
vgg_model = VGG16(include_top=False, input_shape=(224, 224, 3))
fcn_weights = FCN_ResNet50_Weights.DEFAULT
fcn_model = fcn_resnet50(weights=fcn_weights)
fcn_model.eval()



def computeMetrics(img_A, img_B):
  """_summary_

  Args:
      img_A (np.array): The first image
      img_B (np.array): The second image

  Returns:
      np.array: The metrics of the two images  
  """
  input_shape = (224,224,3)

  #resizing our images:
  #img_A_new = cv2.resize(img_A,input_shape[:2]) #assuming (32,32) is the resolution we want
  img_A_new = cv2.resize(img_A, (input_shape[1], input_shape[0]))
  img_B_new = cv2.resize(img_B, (input_shape[1], input_shape[0]))
  #img_B_new = cv2.resize(img_B,input_shape[:2])
  # Check if images loaded successfully
  if img_A_new is None or img_B_new is None:
    print("Error: One or both images could not be loaded. Check paths.")
    # Handle the error, maybe exit or raise an exception
    return
  # Check if images loaded successfully
  

#CS
  features = vgg_model
  model = fcn_model

  #features = model = VGG16(include_top=False, input_shape =input_shape)
  #Only extracts mid to low level features (no deep ones)
  # features = model(inputs=model.input, outputs=model.get_layer("block2_conv2").output)
  
  #preprocesses img_A and img_B to be used by VGG16
  #via converting RBG to BGR, scaling and zero-centering with respected to Imagenet datatset
  pre_A = preprocess_input(img_A_new)
  pre_B = preprocess_input(img_B_new)
  
  batch = np.stack([pre_A,pre_B])
  f_features = features.predict(batch)
  f_A, f_B = f_features[0].flatten(), f_features[1].flatten()
  
  #orthogonal (no similarities)
  cs_result = cosine_similarity([f_A],[f_B])[0][0] 

#CPL
  num = (f_A - f_B)**2 
  cpl_result = np.mean(num)



#kl
  #converts to grayscale
  g_A = cv2.cvtColor(img_A_new, cv2.COLOR_BGR2GRAY)
  g_B = cv2.cvtColor(img_B_new, cv2.COLOR_BGR2GRAY)


  #calculates intensity (in grayscale) of pixel to compare, puts it into a 1D array
  hist_A = cv2.calcHist([g_A], [0], None, [256], [0,256])[:, 0] + 1e-10 #last part added to avoid division by 0 issues.
  hist_B = cv2.calcHist([g_B], [0], None, [256], [0,256])[:, 0] + 1e-10
 
#hist_cmp
  hist_corr = cv2.compareHist(hist_A.reshape(-1,1), hist_B.reshape(-1,1), cv2.HISTCMP_CORREL)
  hist_inter = cv2.compareHist(hist_A.reshape(-1,1), hist_B.reshape(-1,1), cv2.HISTCMP_INTERSECT)


 #normalize the histograms + avoid division by 0 issues
  hist_A = np.clip(hist_A/np.sum(hist_A), 1e-10, None)
  hist_B = np.clip(hist_B/np.sum(hist_B), 1e-10, None)

  kl_result = np.sum(rel_entr(hist_A, hist_B))

#mse
  mse_result = my_mse(img_A_new, img_B_new)

#psnr
  psnr_result = cv2.PSNR(img_A_new, img_B_new)

#ssim 

  '''g_A_float = img_as_float(g_A)
  g_B_float = img_as_float(g_B)

  # For grayscale images
  ssim_result = structural_similarity( g_A_float, g_B_float, data_range=1.0)
 '''

  #for coloured images

  rgb_A = cv2.cvtColor(img_A_new, cv2.COLOR_BGR2RGB)
  rgb_B = cv2.cvtColor(img_B_new, cv2.COLOR_BGR2RGB)
  rgb_A_float = img_as_float(rgb_A)
  rgb_B_float = img_as_float(rgb_B)
  ssim_result = structural_similarity(rgb_A_float, rgb_B_float, data_range=1.0, channel_axis=-1)
  #rgb_A_float = np.clip(rgb_A.astype(np.float32) / 255.0, 0.0, 1.0)
  #where last axis is the colours channel


#sss
  weights = FCN_ResNet50_Weights.DEFAULT
  #model = fcn_resnet50(weights=weights)
  #model.eval()

  #normalizes, resizes and converts images
  '''tensor_A = weights.transforms()(read_image(img_A, mode=ImageReadMode.RGB)).unsqueeze(0)
  tensor_B = weights.transforms()(read_image(img_B, mode=ImageReadMode.RGB)).unsqueeze(0)
'''
  
  #img_A_tensor = torch.from_numpy(img_A_new).permute(2, 0, 1).float() / 255.0
  #img_B_tensor = torch.from_numpy(img_B_new).permute(2, 0, 1).float() / 255.0
  
  img_A_pil = Image.fromarray(img_A_new)
  img_B_pil = Image.fromarray(img_B_new)

  tensor_A = weights.transforms()(img_A_pil).unsqueeze(0)
  tensor_B = weights.transforms()(img_B_pil).unsqueeze(0)
  '''tensor_A = weights.transforms()(read_image(img_A_new, mode=ImageReadMode.RGB)).unsqueeze(0)
  tensor_B = weights.transforms()(read_image(img_B_new, mode=ImageReadMode.RGB)).unsqueeze(0)'''
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
  angles = [0] #can be more, but might need to average them all
  distances = [5] #can be more

  #assuming that distances = [5] (5px apart) and angles = [0] (horizontal to each other)
  glcm_A = compute_glcm_features(img_A_new, distances=distances, angles=angles) 
  glcm_B = compute_glcm_features(img_B_new, distances=distances, angles=angles) 
  #print((glcm_A))

  contr_A = glcm_A[0][0,0]
  contr_B = glcm_B[0][0,0]

  dissim_A = glcm_A[1][0,0]
  dissim_B = glcm_B[1][0,0]

  glcm_contrast = abs(contr_A-contr_B)
  glcm_dissim = abs(dissim_A-dissim_B)

  '''glcm_contrast = abs(graycoprops(glcm_A, 'contrast')[0, 0] - graycoprops(glcm_B, 'contrast')[0, 0])
  glcm_dissim = abs(graycoprops(glcm_A, 'dissimilarity')[0, 0] - graycoprops(glcm_B, 'dissimilarity')[0, 0])
'''
  tsi_result = (glcm_contrast + glcm_dissim)/2

#wd
  wd_result = wasserstein_distance(g_A.flatten(), g_B.flatten())

  metric_names = ["CPL", "CS", "KL", "MSE", "HistCorr", "HistInter", "PSNR", "SSIM", "SSS", "TSI", "WD"]
  metrics = np.array([cpl_result, cs_result, kl_result, mse_result, hist_corr, hist_inter,psnr_result,ssim_result, sss_result, tsi_result, wd_result])
  return metric_names, metrics

def my_mse(img1, img2):
    h, w, c = img1.shape
    err = np.sum(cv2.subtract(img1, img2)**2)
    return err/float(w*h*c)

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
    metrics = (contrast, dissimilarity)
    return metrics
