[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/becausejustyn/xai_ppa/blob/main/notebooks/training.ipynb)


```python
# if running on colab install facenet-pytorch
ON_COLAB = 'google.colab' in str(get_ipython())

if ON_COLAB:
    !pip install -q facenet-pytorch
```


```python
from google.colab import drive

# if running on colab, check if drive is mounted
try:
    with open('/content/drive/My Drive/test.txt') as f:
        print('Google Drive is already mounted.')
except FileNotFoundError:
    drive.mount('/content/drive')
    print('Google Drive has been mounted.')
```

    Google Drive is already mounted.



```python
import os
#import json
#import pickle
#import shutil

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

import torchvision
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize

from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 20)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
```


```python
RANDOM_SEED = 310123
BATCH_SIZE = 128 if torch.cuda.is_available() else 64
torch.manual_seed(RANDOM_SEED)

EPOCHS = 10
LEARNING_RATE = 1e-3

WORKERS = 2 if torch.cuda.is_available() else int(os.cpu_count() / 2) 
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if ON_COLAB:
	BASE_PATH = '/content/drive/MyDrive/xai_faces/'
else:
	BASE_PATH = '../data/'

DARK_UNDERSAMPLED_PATH = BASE_PATH + 'dark_undersampled_cropped' 
LIGHT_UNDERSAMPLED_PATH = BASE_PATH + 'light_undersampled_cropped' 

print(f'Batch Size: {BATCH_SIZE}')
print(f'Workers: {WORKERS}')
print(f'Device: {DEVICE}')
```

    Running on Google Colab
    Batch Size: 128
    Workers: 2
    Device: cuda:0


## Dark Undersampled


```python
transform1 = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

dataset = ImageFolder(DARK_UNDERSAMPLED_PATH, transform = transform1)

# MODEL / resnet
MODEL = InceptionResnetV1(
    classify = True,
    pretrained = 'vggface2',
    num_classes = len(dataset.class_to_idx) 
).to(DEVICE)

OPTIMISER = optim.Adam(MODEL.parameters(), lr = LEARNING_RATE)
SCHEDULER = MultiStepLR(OPTIMISER, [5, 10])
```


```python
# create stratified training split

DARK_DF = pd.read_csv(BASE_PATH + 'dark_undersampled.csv')
LIGHT_DF = pd.read_csv(BASE_PATH + 'light_undersampled.csv')

DARK_DF = DARK_DF.assign(
    image_path_full = lambda x: BASE_PATH + 'dark_undersampled_cropped/' + x['human_id'].astype(str) + '/' + x['render_id'].astype(str) + '.cam_default.f_1.rgb.png',
)

# Group the dataframe by the label
grouped_df = DARK_DF.groupby('skin_labels')

# Calculate the number of instances to sample from each group
group_counts = grouped_df['image_path_full'].count()
sample_counts = (group_counts * 0.8).astype(int)

# Create a list to store the train and validation dataframes
train_dfs = []
val_dfs = []

# Loop through each group and split it into training and validation sets
for name, group in grouped_df:
    group_sample = group.sample(min(len(group), group_counts[name]))
    train_group, val_group = train_test_split(group_sample, test_size = 0.2, random_state = RANDOM_SEED)
    train_dfs.append(train_group)
    val_dfs.append(val_group)

# Concatenate the training and validation dataframes
train_df = pd.concat(train_dfs, ignore_index = True) # light: 0.598465  dark: 0.401535
val_df = pd.concat(val_dfs, ignore_index = True) # light: 0.598465  dark: 0.401535
```

    Running on Google Colab



```python
# Create a list of indices for train and validation datasets
train_indices = [i for i, (path, label) in enumerate(dataset.samples) if path in train_df['image_path_full'].tolist()]
val_indices = [i for i, (path, label) in enumerate(dataset.samples) if path in val_df['image_path_full'].tolist()]

# Use SubsetRandomSampler to specify the indices for the train and validation splits

# Use DataLoader to load the data in batches
train_loader = DataLoader(
    dataset, 
    num_workers = WORKERS, batch_size = BATCH_SIZE,
    sampler = SubsetRandomSampler(train_indices), shuffle = False)

val_loader = DataLoader(
    dataset, 
    num_workers = WORKERS, batch_size = BATCH_SIZE,
    sampler = SubsetRandomSampler(val_indices), shuffle = False)

loss_fn = torch.nn.CrossEntropyLoss()
metrics = {'acc': training.accuracy}
```


```python
writer = SummaryWriter()
writer.iteration = 0
writer.interval = 10

MODEL.eval()

training.pass_epoch(
    MODEL, loss_fn, val_loader, 
    batch_metrics = metrics, show_running = True, 
    device = DEVICE, writer = writer)

for epoch in tqdm(range(EPOCHS)):
  print(f'Epoch {epoch + 1}/{EPOCHS}')
  MODEL.train()
  training.pass_epoch(
      MODEL, loss_fn, train_loader, 
      OPTIMISER, SCHEDULER, 
      batch_metrics = metrics, show_running = True, 
      device = DEVICE, writer = writer)
  MODEL.eval()
  training.pass_epoch(
      MODEL, loss_fn, val_loader, 
      batch_metrics = metrics, show_running = True, 
      device = DEVICE, writer = writer)

writer.close()
```

    Valid |     3/3    | loss:    4.2860 | acc:    0.0106   


      0%|          | 0/10 [00:00<?, ?it/s]

    Epoch 1/10
    Train |    12/12   | loss:    3.4125 | acc:    0.1726   
    Valid |     2/3    | loss:    9.2703 | acc:    0.0430   

     10%|█         | 1/10 [08:34<1:17:11, 514.64s/it]

    Valid |     3/3    | loss:    9.4412 | acc:    0.0371   
    Epoch 2/10
    Train |    12/12   | loss:    1.7189 | acc:    0.5905   
    Valid |     2/3    | loss:    2.7013 | acc:    0.2539   

     20%|██        | 2/10 [08:54<29:48, 223.62s/it]  

    Valid |     3/3    | loss:    2.6693 | acc:    0.2766   
    Epoch 3/10
    Train |    12/12   | loss:    0.7201 | acc:    0.8811   
    Valid |     2/3    | loss:    1.8692 | acc:    0.4844   

     30%|███       | 3/10 [09:12<15:08, 129.78s/it]

    Valid |     3/3    | loss:    1.9134 | acc:    0.4924   
    Epoch 4/10
    Train |    12/12   | loss:    0.2765 | acc:    0.9706   
    Valid |     2/3    | loss:    0.9853 | acc:    0.7656   

     40%|████      | 4/10 [09:30<08:33, 85.57s/it] 

    Valid |     3/3    | loss:    0.9341 | acc:    0.7760   
    Epoch 5/10
    Train |    12/12   | loss:    0.1304 | acc:    0.9906   
    Valid |     2/3    | loss:    0.8354 | acc:    0.7930   

     50%|█████     | 5/10 [09:49<05:07, 61.54s/it]

    Valid |     3/3    | loss:    0.8416 | acc:    0.7914   
    Epoch 6/10
    Train |    12/12   | loss:    0.0532 | acc:    0.9993   
    Valid |     2/3    | loss:    0.6433 | acc:    0.8555   

     60%|██████    | 6/10 [10:07<03:06, 46.69s/it]

    Valid |     3/3    | loss:    0.6053 | acc:    0.8669   
    Epoch 7/10
    Train |    12/12   | loss:    0.0354 | acc:    1.0000   
    Valid |     2/3    | loss:    0.5064 | acc:    0.8750   

     70%|███████   | 7/10 [10:25<01:52, 37.34s/it]

    Valid |     3/3    | loss:    0.5350 | acc:    0.8771   
    Epoch 8/10
    Train |    12/12   | loss:    0.0270 | acc:    1.0000   
    Valid |     2/3    | loss:    0.4067 | acc:    0.9219   

     80%|████████  | 8/10 [10:44<01:02, 31.49s/it]

    Valid |     3/3    | loss:    0.5075 | acc:    0.8942   
    Epoch 9/10
    Train |    12/12   | loss:    0.0236 | acc:    1.0000   
    Valid |     2/3    | loss:    0.4471 | acc:    0.9062   

     90%|█████████ | 9/10 [11:01<00:27, 27.11s/it]

    Valid |     3/3    | loss:    0.4852 | acc:    0.9036   
    Epoch 10/10
    Train |    12/12   | loss:    0.0195 | acc:    1.0000   
    Valid |     2/3    | loss:    0.4647 | acc:    0.9062   

    100%|██████████| 10/10 [11:19<00:00, 67.95s/it]

    Valid |     3/3    | loss:    0.4724 | acc:    0.9036   


    



```python
MODEL_PATH = BASE_PATH + 'models/dark_undersampled1.pt'
#PATH = '/content/drive/MyDrive/xai_faces/models/dark_undersampled1.pt'
STATE_DICT = MODEL.state_dict()

checkpoint = {
    'STATE_DICT': STATE_DICT,
    'TRANSFORMATION': transform1,
    'RANDOM_SEED': RANDOM_SEED,
    'BATCH_SIZE': BATCH_SIZE,
    'EPOCHS': EPOCHS,
    'LEARNING_RATE': LEARNING_RATE,
    'WORKERS': WORKERS,
    'DEVICE': DEVICE
}

torch.save(checkpoint, MODEL_PATH)
```

## Light Undersampled


```python
# so I do not need to restart the runtime
del MODEL, writer, train_loader, val_loader, dataset
```


```python
# load the dataset without a transformation 
dataset = ImageFolder(LIGHT_UNDERSAMPLED_PATH, transform = transform1)

# MODEL / resnet
MODEL = InceptionResnetV1(
    classify = True,
    pretrained = 'vggface2',
    num_classes = len(dataset.class_to_idx) 
).to(DEVICE)

OPTIMISER = optim.Adam(MODEL.parameters(), lr = LEARNING_RATE)
SCHEDULER = MultiStepLR(OPTIMISER, [5, 10])
```


```python
# create stratified training split

LIGHT_DF = LIGHT_DF.assign(
    image_path_full = lambda x: BASE_PATH + 'light_undersampled_cropped/' + x['human_id'].astype(str) + '/' + x['render_id'].astype(str) + '.cam_default.f_1.rgb.png',
)

# Group the dataframe by the label
grouped_df = LIGHT_DF.groupby('skin_labels')

# Calculate the number of instances to sample from each group
group_counts = grouped_df['image_path_full'].count()
sample_counts = (group_counts * 0.8).astype(int)

# Create a list to store the train and validation dataframes
train_dfs = []
val_dfs = []

# Loop through each group and split it into training and validation sets
for name, group in grouped_df:
    group_sample = group.sample(min(len(group), group_counts[name]))
    train_group, val_group = train_test_split(group_sample, test_size = 0.2, random_state = RANDOM_SEED)
    train_dfs.append(train_group)
    val_dfs.append(val_group)

# Concatenate the training and validation dataframes
train_df = pd.concat(train_dfs, ignore_index = True) # light: 0.598465  dark: 0.401535
val_df = pd.concat(val_dfs, ignore_index = True) # light: 0.598465  dark: 0.401535

# Create a list of indices for train and validation datasets
train_indices = [i for i, (path, label) in enumerate(dataset.samples) if path in train_df['image_path_full'].tolist()]
val_indices = [i for i, (path, label) in enumerate(dataset.samples) if path in val_df['image_path_full'].tolist()]

# Use SubsetRandomSampler to specify the indices for the train and validation splits

# Use DataLoader to load the data in batches
train_loader = DataLoader(
    dataset, 
    num_workers = WORKERS, batch_size = BATCH_SIZE,
    sampler = SubsetRandomSampler(train_indices), shuffle = False)

val_loader = DataLoader(
    dataset, 
    num_workers = WORKERS, batch_size = BATCH_SIZE,
    sampler = SubsetRandomSampler(val_indices), shuffle = False)

loss_fn = torch.nn.CrossEntropyLoss()
metrics = {'acc': training.accuracy}
```


```python
writer = SummaryWriter()
writer.iteration = 0
writer.interval = 10

MODEL.eval()

training.pass_epoch(
    MODEL, loss_fn, val_loader, 
    batch_metrics = metrics, show_running = True, 
    device = DEVICE, writer = writer)

for epoch in tqdm(range(EPOCHS)):
  print(f'Epoch {epoch + 1}/{EPOCHS}')
  MODEL.train()
  training.pass_epoch(
      MODEL, loss_fn, train_loader, 
      OPTIMISER, SCHEDULER, 
      batch_metrics = metrics, show_running = True, 
      device = DEVICE, writer = writer)
  MODEL.eval()
  training.pass_epoch(
      MODEL, loss_fn, val_loader, 
      batch_metrics = metrics, show_running = True, 
      device = DEVICE, writer = writer)

writer.close()
```

    Valid |     3/3    | loss:    4.3050 | acc:    0.0057   


      0%|          | 0/10 [00:00<?, ?it/s]

    Epoch 1/10
    Train |    12/12   | loss:    3.5581 | acc:    0.1258   
    Valid |     2/3    | loss:   12.3821 | acc:    0.0469   

     10%|█         | 1/10 [09:33<1:26:05, 573.99s/it]

    Valid |     3/3    | loss:   13.0821 | acc:    0.0312   
    Epoch 2/10
    Train |    12/12   | loss:    1.9645 | acc:    0.5153   
    Valid |     2/3    | loss:    3.2770 | acc:    0.1445   

     20%|██        | 2/10 [09:52<32:55, 246.95s/it]  

    Valid |     3/3    | loss:    3.2681 | acc:    0.1498   
    Epoch 3/10
    Train |    12/12   | loss:    1.0584 | acc:    0.7624   
    Valid |     2/3    | loss:    2.5067 | acc:    0.3164   

     30%|███       | 3/10 [10:10<16:39, 142.83s/it]

    Valid |     3/3    | loss:    2.5097 | acc:    0.3021   
    Epoch 4/10
    Train |    12/12   | loss:    0.5408 | acc:    0.9026   
    Valid |     2/3    | loss:    1.6707 | acc:    0.5430   

     40%|████      | 4/10 [10:27<09:18, 93.16s/it] 

    Valid |     3/3    | loss:    1.6627 | acc:    0.5507   
    Epoch 5/10
    Train |    12/12   | loss:    0.2557 | acc:    0.9638   
    Valid |     2/3    | loss:    1.2124 | acc:    0.6836   

     50%|█████     | 5/10 [10:45<05:28, 65.79s/it]

    Valid |     3/3    | loss:    1.2257 | acc:    0.6790   
    Epoch 6/10
    Train |    12/12   | loss:    0.1066 | acc:    0.9967   
    Valid |     2/3    | loss:    0.5746 | acc:    0.8594   

     60%|██████    | 6/10 [11:03<03:17, 49.47s/it]

    Valid |     3/3    | loss:    0.6334 | acc:    0.8559   
    Epoch 7/10
    Train |    12/12   | loss:    0.0767 | acc:    0.9980   
    Valid |     2/3    | loss:    0.5632 | acc:    0.8867   

     70%|███████   | 7/10 [11:21<01:58, 39.41s/it]

    Valid |     3/3    | loss:    0.5360 | acc:    0.8836   
    Epoch 8/10
    Train |    12/12   | loss:    0.0510 | acc:    0.9987   
    Valid |     2/3    | loss:    0.4973 | acc:    0.8789   

     80%|████████  | 8/10 [11:39<01:04, 32.42s/it]

    Valid |     3/3    | loss:    0.5220 | acc:    0.8847   
    Epoch 9/10
    Train |    12/12   | loss:    0.0429 | acc:    0.9993   
    Valid |     2/3    | loss:    0.5308 | acc:    0.8867   

     90%|█████████ | 9/10 [11:56<00:27, 27.75s/it]

    Valid |     3/3    | loss:    0.4826 | acc:    0.9025   
    Epoch 10/10
    Train |    12/12   | loss:    0.0354 | acc:    1.0000   
    Valid |     2/3    | loss:    0.4547 | acc:    0.8945   

    100%|██████████| 10/10 [12:14<00:00, 73.49s/it]

    Valid |     3/3    | loss:    0.4719 | acc:    0.9014   


    



```python
MODEL_PATH = BASE_PATH + 'models/light_undersampled1.pt'
#PATH = '/content/drive/MyDrive/xai_faces/models/light_undersampled1.pt'
STATE_DICT = MODEL.state_dict()

checkpoint = {
    'STATE_DICT': STATE_DICT,
    'TRANSFORMATION': transform1,
    'RANDOM_SEED': RANDOM_SEED,
    'BATCH_SIZE': BATCH_SIZE,
    'EPOCHS': EPOCHS,
    'LEARNING_RATE': LEARNING_RATE,
    'WORKERS': WORKERS,
    'DEVICE': DEVICE
}

torch.save(checkpoint, MODEL_PATH)
```

```python
checkpoint = torch.load(PATH)
MODEL.load_state_dict(checkpoint['STATE_DICT'])

RANDOM_SEED = checkpoint['RANDOM_SEED']
BATCH_SIZE = checkpoint['BATCH_SIZE']
EPOCHS = checkpoint['EPOCHS']
LEARNING_RATE = checkpoint['LEARNING_RATE']
WORKERS = checkpoint['WORKERS']
DEVICE = checkpoint['DEVICE']
```
