```python
# if running on colab install facenet-pytorch
ON_COLAB = 'google.colab' in str(get_ipython())

if ON_COLAB:
    !pip install -q facenet-pytorch zennit

if ON_COLAB:
    BASE_PATH = '/content/drive/MyDrive/xai_faces/'
    MODEL_PATH = '/content/drive/MyDrive/xai_faces/models/'
else:
    BASE_PATH = '../data/'
    MODEL_PATH = '../models/'
        
DARK_UNDERSAMPLED_PATH = BASE_PATH + 'dark_undersampled_cropped' 
LIGHT_UNDERSAMPLED_PATH = BASE_PATH + 'light_undersampled_cropped' 
FAIR_PATH = BASE_PATH + 'diverse_human_faces_cropped'
DARK_MODEL_PATH = MODEL_PATH + 'dark_undersampled1.pt'
LIGHT_MODEL_PATH = MODEL_PATH + 'light_undersampled1.pt'
FAIR_MODEL_PATH = MODEL_PATH + 'fair.pt'

RANDOM_SEED = 80223
```


```python
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch.nn import Linear, functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor, Normalize, Compose, Resize, CenterCrop

from facenet_pytorch import InceptionResnetV1

from zennit.image import imgify, imsave
from zennit.attribution import Gradient, SmoothGrad
from zennit.composites import EpsilonGammaBox, EpsilonPlusFlat, SpecialFirstLayerMapComposite
from zennit.torchvision import ResNetCanonizer

from zennit.rules import Epsilon, ZPlus, ZBox, Norm, Pass, Flat
from zennit.types import Convolution, Activation, AvgPool, BatchNorm, MaxPool, Linear as AnyLinear

import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
coldnhot = '0ff, 00f, 80:000, f00, ff0, fff'
```


```python
def find_index(dataset, image_path):
    for i, (path, target) in enumerate(dataset.imgs):
        if path == image_path:
            return i
    raise ValueError("Image path not found in the dataset")

def get_image_data(dataset, index, transform, class_to_idx):
    input_image, input_target = dataset.imgs[index]
    image = Image.open(input_image)
    data = transform(image)[None]
    target = torch.eye(len(class_to_idx))[[input_target]]
    return input_image, input_target, image, data, target

def generate_xai_image(method, model, data, target, noise_level = 0.1, n_iter = 20, symmetric = False, cmap = 'hot'):
    if method == 'Gradient':
        with Gradient(model = model) as attributor: 
            output, attribution = attributor(data, target)
    elif method == 'SmoothGrad':
        with SmoothGrad(noise_level = noise_level, n_iter = n_iter, model = model) as attributor: 
            output, attribution = attributor(data, target)
    ### Layer-wise Relevance Propagation (LRP) with EpsilonPlusFlat
    elif method == 'EpsilonPlusFlat':
        composite = EpsilonPlusFlat()
        with Gradient(model = model, composite = composite) as attributor: 
            output, attribution = attributor(data, target)
    ### LRP with EpsilonGammaBox
    elif method == 'EpsilonGammaBox':
        # the EpsilonGammaBox composite needs the lowest and highest values, which are here for ImageNet 0. and 1. with a different normalization for each channel
        transform_norm = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        low, high = transform_norm(torch.tensor([[[[[0.]]] * 3], [[[[1.]]] * 3]])) # create a composite, specifying required arguments
        composite = EpsilonGammaBox(low = low, high = high)
        with Gradient(model = model, composite = composite) as attributor: 
            output, attribution = attributor(data, target)
    else:
        raise ValueError("Invalid method name. Choose either 'Gradient', 'SmoothGrad', 'EpsilonPlusFlat' or 'EpsilonGammaBox'.")
    
### Wrapper functions for LRP

# EpsilonPlusFlat
def lrp_plus_flat(model, data, target, cmap = 'coldnhot', image_name = None, save_image = False, symmetric = False, level = 1.):
  # use the ResNet-specific canonizer
  canonizer = ResNetCanonizer()

  # the ZBox rule needs the lowest and highest values, which are here for ImageNet 0. and 1. with a different normalization for each channel        
  transform_norm = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
  low, high = transform_norm(torch.tensor([[[[[0.]]] * 3], [[[[1.]]] * 3]]))

  # create a composite, specifying the canonizers
  composite = EpsilonPlusFlat(canonizers = [canonizer])
  with Gradient(model = model, composite = composite) as attributor:
    output, attribution = attributor(data, target)

  relevance = attribution.sum(1)
  # create an image of the visualize attribution the relevance is only positive, so we use symmetric=False and an unsigned color-map
  if save_image:
    if image_name is None:
      raise ValueError("Please provide an image name")
    imsave(image_name, relevance, symmetric = symmetric, cmap = cmap, level = level)
  else:
    img = imgify(relevance, symmetric = symmetric, cmap = cmap, level = level)
    return relevance, output, img

# LRP with EpsilonGammaBox
def lrp_eps_gamma(model, data, target, cmap = 'coldnhot', image_name = None, save_image = False, symmetric = False, level = 1.):
  # use the ResNet-specific canonizer
  canonizer = ResNetCanonizer()

  # the ZBox rule needs the lowest and highest values, which are here for ImageNet 0. and 1. with a different normalization for each channel        
  transform_norm = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
  low, high = transform_norm(torch.tensor([[[[[0.]]] * 3], [[[[1.]]] * 3]]))

  # create a composite, specifying the canonizers
  composite = EpsilonGammaBox(low = low, high = high, canonizers = [canonizer])
  with Gradient(model = model, composite = composite) as attributor:
    output, attribution = attributor(data, target)

  relevance = attribution.sum(1)
  # create an image of the visualize attribution the relevance is only positive, so we use symmetric=False and an unsigned color-map
  if save_image:
    if image_name is None:
      raise ValueError("Please provide an image name")
    imsave(image_name, relevance, symmetric = symmetric, cmap = cmap, level = level)
  else:
    img = imgify(relevance, symmetric = symmetric, cmap = cmap, level = level)
    return relevance, output, img

# LRP with custom LayerMapComposite
def layer_map_comp(model, data, target, cmap = 'coldnhot', image_name = None, save_image = False, symmetric = False, level = 1.):
  # use the ResNet-specific canonizer
  canonizer = ResNetCanonizer()

  # the ZBox rule needs the lowest and highest values, which are here for ImageNet 0. and 1. with a different normalization for each channel        
  transform_norm = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
  low, high = transform_norm(torch.tensor([[[[[0.]]] * 3], [[[[1.]]] * 3]]))

  # create a composite, specifying the canonizers, if any
  composite = SpecialFirstLayerMapComposite(
            # the layer map is a list of tuples, where the first element is the target
            # layer type, and the second is the rule template
    	layer_map = [
        	(Activation, Pass()),  # ignore activations
        	(AvgPool, Norm()),  # normalize relevance for any AvgPool
        	(Convolution, ZPlus()),  # any convolutional layer
        	(Linear, Epsilon(epsilon = 1e-6)),  # this is the dense Linear, not any
        	(BatchNorm, Pass()),  # ignore BatchNorm
        	],
    	# the first map is only used once, to the first module which applies to the
    	# map, i.e. here the first layer of type AnyLinear
    	first_map = [(AnyLinear, ZBox(low, high))]
    	)
  with Gradient(model = model, composite = composite) as attributor:
    output, attribution = attributor(data, target)

  relevance = attribution.sum(1)
  # create an image of the visualize attribution the relevance is only positive, so we use symmetric=False and an unsigned color-map
  if save_image:
    if image_name is None:
      raise ValueError("Please provide an image name")
    imsave(image_name, relevance, symmetric = symmetric, cmap = cmap, level = level)
  else:
    img = imgify(relevance, symmetric = symmetric, cmap = cmap, level = level)
    return relevance, output, img
  
# SmoothGrad wrapper
def xai_sg(model, data, target, cmap = 'coldnhot', image_name = None, save_image = False, symmetric = False, level = 1., n_iter = 20, noise_level = 0.1):

    with SmoothGrad(model = model, n_iter = n_iter, noise_level = noise_level) as attributor:
        output, attribution = attributor(data, target)
    
    relevance = attribution.abs().sum(1)
    if save_image:
        if image_name is None:
            raise ValueError("Please provide an image name")
        imsave(image_name, relevance, symmetric = symmetric, cmap = cmap, level = level)
    else:
        img = imgify(relevance, symmetric = symmetric, cmap = cmap, level = level)
        return relevance, output, img
        
def get_index(file_path):
    for i, (image_path, _) in enumerate(dataset.imgs):
        if image_path == file_path:
            return i
    return -1

def get_pred(model, data, target, dataset):
    # create the attributor, specifying model
    with Gradient(model = model) as attributor:
        # compute the model output, do not need the attribution
        output, _ = attributor(data, target)

    class_idx = dataset.class_to_idx
    class_to_name = {class_idx: class_name for (class_name, class_idx) in class_idx.items()}
    predicted_idx = output.argmax(1)[0].item()
    predicted_class = class_to_name[predicted_idx]
    predicted_prob = F.softmax(output, dim=1)[0][predicted_idx].item()
    return predicted_class, predicted_prob

def get_prediction(row):
    input_image, input_target = dataset.imgs[row['dataset_index']]
    image = Image.open(input_image)
    data = transform(image)[None]
    target = torch.eye(len(dataset.classes))[[input_target]]
    predicted_class, predicted_prob = get_pred(model=MODEL, data=data, target=target, dataset=dataset)
    return int(predicted_class), predicted_prob
```

```bash
# Note use this from the terminal, which will use the base path

# create varoab;e
export experiment1_path=xai_samples

for i in {1..5}; do
    for heatmap in cold hot coldnhot; do
        mkdir -p $experiment1_path/XAI$i/$heatmap/dark_undersampled \
                 $experiment1_path/XAI$i/$heatmap/light_undersampled \
                 $experiment1_path/XAI$i/$heatmap/fair_model
    done
done

# check that they were made
for i in {1..5}; do
    echo "Subdirectories in XAI$i:"
    for heatmap in cold hot coldnhot; do
        ls $experiment1_path/XAI$i/$heatmap | grep -E 'dark_undersampled|light_undersampled|fair_model'
    done
done
```


```python
transform_norm = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# define the full tensor transform
transform = Compose([
    base_transform,
    ToTensor(),
    transform_norm,
])
```


```python
# transformations

base_transform = Compose([CenterCrop(224)])
transform_norm = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

transform = Compose([
    base_transform,
    ToTensor(),
    transform_norm,
])

#dataset = ImageFolder(FULL_DATA_PATH, transform = base_transform)
dataset = ImageFolder(FULL_DATA_PATH)
class_to_idx = dataset.class_to_idx

# MODEL / resnet
MODEL = InceptionResnetV1(
    classify = True,
    num_classes = len(class_to_idx) 
)

#checkpoint = torch.load(FAIR_MODEL_PATH, map_location = torch.device('cpu'))
checkpoint = torch.load(FAIR_MODEL_PATH, map_location = DEVICE)
MODEL.load_state_dict(checkpoint['STATE_DICT'])
MODEL.to(DEVICE)
MODEL.eval();
```

## Fair Model


```python
### XAI 1: Epsilon Plus Flat
for index, row in tqdm(fair_samples.iterrows(), total = len(fair_samples)):
  input_image, input_target = dataset.imgs[row['dataset_index']]
  image_name = f"xai_{row['image_path'].split('/')[-1]}"
  #image_path = os.path.join(XAI1_PATH, 'hot', 'fair_model', image_name)
  image_path = os.path.join(XAI1_PATH, 'coldnhot', 'fair_model', image_name)
  image = Image.open(input_image)
  data = transform(image)[None]
  target = torch.eye(len(class_to_idx))[[input_target]]
  #lrp_plus_flat(model = MODEL, data = data, target = target, save_image = True, image_name = image_path, symmetric = False, cmap = hot)
  lrp_plus_flat(model = MODEL, data = data, target = target, save_image = True, image_name = image_path, symmetric = True, cmap = coldnhot)
```


```python
### XAI 2: Epsilon Gamma Box
for index, row in tqdm(fair_samples.iterrows(), total = len(fair_samples)):
  input_image, input_target = dataset.imgs[row['dataset_index']]
  image_name = f"xai_{row['image_path'].split('/')[-1]}"
  # image_name = f"xai_{row['image_path'].split('/')[-1]}"
  #image_path = os.path.join(XAI2_PATH, 'hot', 'fair_model', image_name)
  image_path = os.path.join(XAI2_PATH, 'coldnhot', 'fair_model', image_name)
  image = Image.open(input_image)
  data = transform(image)[None]
  target = torch.eye(len(class_to_idx))[[input_target]]
  lrp_eps_gamma(model = MODEL, data = data, target = target, save_image = True, image_name = image_path, symmetric = True, cmap = coldnhot)
  #lrp_eps_gamma(model = MODEL, data = data, target = target, save_image = True, image_name = image_path, symmetric = False, cmap = hot)
```


```python
### XAI 3: Layer Map Composite
for index, row in tqdm(fair_samples.iterrows(), total = len(fair_samples)):
  input_image, input_target = dataset.imgs[row['dataset_index']]
  image_name = f"xai_{row['image_path'].split('/')[-1]}"
  image_path = os.path.join(XAI3_PATH, 'hot', 'fair_model', image_name)
  #image_path = os.path.join(XAI3_PATH, 'coldnhot', 'fair_model', image_name)
  image = Image.open(input_image)
  data = transform(image)[None]
  target = torch.eye(len(class_to_idx))[[input_target]]
  #layer_map_comp(model = MODEL, data = data, target = target, save_image = True, image_name = image_path, symmetric = True, cmap = coldnhot)
  layer_map_comp(model = MODEL, data = data, target = target, save_image = True, image_name = image_path, symmetric = False, cmap = hot)
```


```python

```


```python

```


```python

```


```python

```
