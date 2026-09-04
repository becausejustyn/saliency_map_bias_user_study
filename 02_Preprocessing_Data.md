This will create a new directory for `data/dark_undersampled/` and `data/light_undersampled/` that will be cropped and downsized to `(256, 256)`. The file size is much smaller once resized so I am not too concerned.


```python
ON_COLAB = 'google.colab' in str(get_ipython())

if ON_COLAB:
    !pip install -q facenet_pytorch
```


```python
# if running on colab, mount google drive

if ON_COLAB:
    from google.colab import drive

    # checking if drive is mounted
    try:
        with open('/content/drive/My Drive/test.txt') as f:
            print('Google Drive is already mounted.')
    except FileNotFoundError:
        drive.mount('/content/drive')
        print('Google Drive has been mounted.')
```


```python
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

import torchvision
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize

from sklearn.model_selection import train_test_split
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training

import matplotlib.pyplot as plt
```


```python
# check that numpy is using 1.21.6 or MTCNN will not work
!pip show numpy | grep 'Version:' | cut -d ' ' -f 2
```

    1.21.6



```python
ON_COLAB = 'google.colab' in str(get_ipython())
RANDOM_SEED = 310123
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

EPOCHS = 10
LEARNING_RATE = 1e-3

WORKERS = int(os.cpu_count() / 2) 
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if ON_COLAB:
    print("Running on Google Colab")
    DARK_UNDERSAMPLED_PATH = '/content/drive/MyDrive/xai_faces/dark_undersampled' # '/content/drive/MyDrive/xai_faces/dark_undersampled_abridged_cropped'
    LIGHT_UNDERSAMPLED_PATH = '/content/drive/MyDrive/xai_faces/light_undersampled' # '/content/drive/MyDrive/xai_faces/light_undersampled_abridged_cropped'
else:
    print("Not running on Google Colab")
    DARK_UNDERSAMPLED_PATH = '../data/dark_undersampled' # 'data/dark_undersampled_abridged_cropped'
    LIGHT_UNDERSAMPLED_PATH = '../data/light_undersampled' # 'data/light_undersampled_abridged_cropped/'
```


```python
print(f'Batch Size: {BATCH_SIZE}')
print(f'Workers: {WORKERS}')
print(f'Device: {DEVICE}')
```

## Cropping Images


```python
NEW_SIZE = (512, 512)
CROP_SIZE = 256

# help(MTCNN)

# most of these values are defaults, but I'm including them here for clarity
mtcnn = MTCNN(
    image_size = CROP_SIZE, 
    margin = 0, 
    in_face_size = 20, 
    thresholds = [0.6, 0.7, 0.7], 
    factor = 0.709, 
    post_process = True, 
    select_largest = True, 
    device = DEVICE) # image_size = 160
```

### Dark Undersampled


```python
dataset = ImageFolder(DARK_UNDERSAMPLED_PATH, transform = Resize(NEW_SIZE))

dataset.samples = [
    (p, p.replace(DARK_UNDERSAMPLED_PATH, DARK_UNDERSAMPLED_PATH + '_cropped'))
        for p, _ in dataset.samples
]

loader = DataLoader(
    dataset,
    num_workers = WORKERS,
    batch_size = BATCH_SIZE,
    collate_fn = training.collate_pil
)

# prevent warnings from cropping images
with np.testing.suppress_warnings() as sup:
    sup.filter(category = np.VisibleDeprecationWarning)
    for i, (x, y) in tqdm(enumerate(loader), total = len(loader)):
        mtcnn(x, save_path = y)
```

### Light Undersampled


```python
dataset = ImageFolder(LIGHT_UNDERSAMPLED_PATH, transform = Resize(NEW_SIZE))

dataset.samples = [
    (p, p.replace(LIGHT_UNDERSAMPLED_PATH, LIGHT_UNDERSAMPLED_PATH + '_cropped'))
        for p, _ in dataset.samples
]

loader = DataLoader(
    dataset,
    num_workers = WORKERS,
    batch_size = BATCH_SIZE,
    collate_fn = training.collate_pil
)

# prevent warnings from cropping images
with np.testing.suppress_warnings() as sup:
    sup.filter(category = np.VisibleDeprecationWarning)
    for i, (x, y) in tqdm(enumerate(loader), total = len(loader)):
        mtcnn(x, save_path = y)
```

## Creating Train/Val Splits


```python
if ON_COLAB:
    print("Running on Google Colab")
    DARK_UNDERSAMPLED_PATH = '/content/drive/MyDrive/xai_faces/dark_undersampled.csv'
    LIGHT_UNDERSAMPLED_PATH = '/content/drive/MyDrive/xai_faces/light_undersampled.csv' 
else:
    print("Not running on Google Colab")
    DARK_UNDERSAMPLED_PATH = '../data/dark_undersampled.csv'
    LIGHT_UNDERSAMPLED_PATH = '../data/light_undersampled.csv' 
```

    Not running on Google Colab



```python
dark_undersampled_df = pd.read_csv(DARK_UNDERSAMPLED_PATH)

# if ran locally, path requires '../data/' prefix
dark_undersampled_df = dark_undersampled_df.assign(
    image_path_full = lambda x: 'dark_undersampled/' + x['human_id'].astype(str) + '/' + x['render_id'].astype(str) + '.cam_default.f_1.rgb.png',
)

# Group the dataframe by the label
dark_grouped_df = dark_undersampled_df.groupby('skin_labels')

# Calculate the number of instances to sample from each group
dark_group_counts = dark_grouped_df['image_path_full'].count()
dark_sample_counts = (dark_group_counts * 0.8).astype(int)

# Create a list to store the train and validation dataframes
train_dfs, val_dfs = [], []

# Loop through each group and split it into training and validation sets
for name, group in dark_grouped_df:
    group_sample = group.sample(min(len(group), dark_group_counts[name]), random_state = RANDOM_SEED)
    train_group, val_group = train_test_split(group_sample, test_size = 0.2)
    train_dfs.append(train_group)
    val_dfs.append(val_group)

# Concatenate the training and validation dataframes
dark_undersampled_train_idx = pd.concat(train_dfs, ignore_index = True) # light: 0.598465  dark: 0.401535
dark_undersampled_val_idx = pd.concat(val_dfs, ignore_index = True) # light: 0.598465  dark: 0.401535

dark_undersampled_train_idx.to_csv('../data/dark_train_split.csv', index = False)
dark_undersampled_val_idx.to_csv('../data/dark_val_split.csv', index = False)
```


```python
light_undersampled_df = pd.read_csv(LIGHT_UNDERSAMPLED_PATH)

# if ran locally, path requires '../data/' prefix
light_undersampled_df = light_undersampled_df.assign(
    image_path_full = lambda x: 'light_undersampled/' + x['human_id'].astype(str) + '/' + x['render_id'].astype(str) + '.cam_default.f_1.rgb.png',
)

# Group the dataframe by the label
light_grouped_df = light_undersampled_df.groupby('skin_labels')

# Calculate the number of instances to sample from each group
light_group_counts = light_grouped_df['image_path_full'].count()
light_sample_counts = (light_group_counts * 0.8).astype(int)

# Create a list to store the train and validation dataframes
train_dfs, val_dfs = [], []

# Loop through each group and split it into training and validation sets
for name, group in light_grouped_df:
    group_sample = group.sample(min(len(group), light_group_counts[name]), random_state = RANDOM_SEED)
    train_group, val_group = train_test_split(group_sample, test_size = 0.2)
    train_dfs.append(train_group)
    val_dfs.append(val_group)

# Concatenate the training and validation dataframes
light_undersampled_train_idx = pd.concat(train_dfs, ignore_index = True) # light: 0.537542  dark: 0.462458
light_undersampled_val_idx = pd.concat(val_dfs, ignore_index = True) # light: 0.538462  dark: 0.461538

light_undersampled_train_idx.to_csv('../data/light_train_split.csv', index = False)
light_undersampled_val_idx.to_csv('../data/light_val_split.csv', index = False)
```
