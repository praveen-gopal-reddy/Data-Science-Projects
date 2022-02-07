"""

Template code for COMP5623M CW1 Question 2

python explore.py --image_path XX --use_pre_trained True


"""

import argparse
import torch
import numpy as np
import torchvision
from torch import nn, optim
import torchvision.transforms as transforms
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image



# Set up training arguments and parse
parser = argparse.ArgumentParser(description='Explore pre-trained AlexNet')


parser.add_argument(
    '--image_path', type=str,
    help='Full path to the input image to load.')
parser.add_argument(
    '--use_pre_trained', type=bool, default=True,
    help='Load pre-trained weights?')


args = parser.parse_args()

# Device configuration - defaults to CPU unless GPU is available on device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


print("=======================================")
print("                PARAMS               ")
print("=======================================")
for arg in vars(args):
    print(F"{arg:>20} {getattr(args, arg)}")


#########################################################################
#
#        QUESTION 2.1.2 code here
# 
#########################################################################


# Read in image located at args.image_path


image_x = Image.open(args.image_path)

# Normalizations expected by pre-trained net, to apply in the image transform
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
transform_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

#apply image transformation to image_x
image_x = image_x.convert('RGB')
image_x = transform_image(image_x) 
image_x = image_x.unsqueeze(0)



# Loads the model and downloads pre-trained weights if not already downloaded
model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

# To see the AlexNet architecture
print(model)

model.eval()


# Pass image through a single forward pass of the network
def image_forward_pass(model):
    with torch.no_grad():
        outputs = model(image_x)
        _, predicted = torch.max(outputs.data, 1)
    return predicted

#call imgage_forward_pass function
image_x_fp = image_forward_pass(model)
print(image_x_fp)






# layer indices of each conv layer in AlexNet
conv_layer_indices = [0, 3, 6, 8, 10]

#########################################################################
#
#        QUESTION 2.1.3 
# 
#########################################################################

def extract_filter(conv_layer_idx, model):
        # Extracts a single filter from the specified convolutional layer,
		# zero-indexed where 0 indicates the first conv layer.
 
		# Args:
			# conv_layer_idx (int): index of convolutional layer
			# model (nn.Module): PyTorch model to extract from
	# Extract filter
    the_filter = model.features[conv_layer_idx].weight.data
    return the_filter

#loop convolutional layers indices or pass specific index in extract filter function  
for ind in conv_layer_indices:
    f_weights = extract_filter(ind,model)
    print("Filters_all:", f_weights)
    #print("Filters_all_shape:", f_weights.shape)	
      
#########################################################################
#
#        QUESTION 2.1.4
# 
#########################################################################

def extract_feature_maps_all(input, model):
	# Extract all feature maps
	# Hint: use conv_layer_indices to access
    #get indices after conv+relu activation as per question asked
    conv_relu_activation=[1,4,7,9,11]	
    image_x = input
    feature_maps = []
    features = model.features
    with torch.no_grad():
        for index,layer in enumerate(features):
            image_x = layer(image_x)
            if index in conv_relu_activation:
                feature_maps.append(image_x)
    return feature_maps
feature_map_all = extract_feature_maps_all(image_x, model)
print("feature_maps_all:", feature_map_all)
	

###########################################################################
#
#       2.2.1 visualization code for filters and features map
#
#
###########################################################################

#function to get feature map at specific layer say 0

def extract_feature_maps_specific_layer(input, model,specific_index):
    image_x = input
    features = model.features
    with torch.no_grad():
        for index,layer in enumerate(features):
            image_x = layer(image_x)
            if specific_index == index:
                return image_x

specific_layer_index = 0                 
#pass specific_layer_index values manually to see feature maps according to layer

single_feature_map = extract_feature_maps_specific_layer(image_x, model, specific_layer_index)
print("single map:", single_feature_map)

#plot feature map at specific layer

def visualize_feature_map(f_map):
    #print(f_map.shape)
    f_map = f_map.squeeze(0)
    f_map = f_map.numpy()
    f_map_num = f_map.shape[0]
    row_num = np.ceil(np.sqrt(f_map_num))
    plt.figure(figsize=(20, 17))
    for index in range(1, f_map_num+1):
        plt.subplot(20, 20, index)
        plt.imshow(f_map[index-1], cmap='gray')
        plt.axis('off')
    plt.show()

#invoke the function	
visualize_feature_map(single_feature_map)

#plot filters
def visualize_filter(filter_weight_specific_index):
    plt.figure(figsize=(20, 17))
    for i, filter in enumerate(filter_weight_specific_index):
        plt.subplot(20, 20, i+1)
        plt.imshow(filter[0, :, :].detach(), cmap='gray')
        plt.axis('off')
    plt.show()
 

#invoke the extract filter function and visualize_filter function
# here parameter filter_layer_index need to be changed manually i,e layer index.
# example 
""" extract_filter(0,model)
"""

filter_weight_specific_index =extract_filter(0,model)   # pass filter_layer_index values manually according to layer index
visualize_filter(filter_weight_specific_index)
print("single filter:",filter_weight_specific_index)
   









