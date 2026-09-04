```python
import os
import re
import json
import shutil
from tqdm import tqdm

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

RANDOM_SEED = 310123
N_TOTAL = 30
```


```python
# move the jsonl file out of the diverse_human_faces folder
FILE_PATH = 'data/diverse_human_faces/metadata.jsonl'
DEST_PATH = 'data/metadata.jsonl'

shutil.move(FILE_PATH, DEST_PATH)
```


```python
# for some reason, python does not unnest the JSONL file correctly, so here is a script to have it done in R
# it will return metadata.parquet and metadata.csv
# then move it to the data directory
!Rscript ../src/json_convert.R data/metadata.jsonl
```


```python
def get_image_path(human_id: int, render_id: int):
    return f'{human_id:.0f}/{render_id:.0f}.cam_default.f_1.rgb.png' 

def get_full_image_path(folder, human_id: int, render_id: int):
    return f'{folder}/{human_id:.0f}/{render_id:.0f}.cam_default.f_1.rgb.png' 
```


```python
# using parquet to speed up the loading process
# you can do this straight with pandas, but it is slower
df = pq.read_table('../data/metadata.parquet').to_pandas()

# df.filter(like = 'id').columns
# render_id is the same as the folder number I created, eg. render_id 0 == file_0
# might be worthwhile to add .zfill() to the render_id column/folder_name

human_info = df.rename(columns = {
    'scene.identity_metadata.id': 'human_id', 
    'scene.identity_metadata.sex' : 'sex', 
    'scene.identity_metadata.age' : 'age',
    'scene.identity_metadata.ethnicity': 'ethnicity',
    'scene.facial_attributes.head_turn.yaw' : 'yaw',
    'scene.facial_attributes.head_turn.roll' : 'roll',
    'scene.facial_attributes.head_turn.pitch' : 'pitch',
    'scene.identity_metadata.skin_tone' : 'skin_tone',
    }, inplace = False).assign(
        sex = lambda x: np.where(x['sex'] == 'female', 1, 0),
    )[[
        'human_id', 'render_id', 'task_id', 'sex', 'age', 'ethnicity', 'yaw', 'roll', 'pitch', 'skin_tone'
    ]]

# get list of column names to change to float
# cols_to_change = human_info.filter(regex = '(^(?!.*_id))').columns.tolist()
cols_to_change = [col for col in human_info.columns.tolist() if col not in ['ethnicity', 'task_id', 'render_id']]
human_info[cols_to_change] = human_info[cols_to_change].astype(float)

human_info = human_info.assign(
    # make an int type for the human_id
    human_id = lambda x: x['human_id'].astype(int),
    # if you only want 3 directions -> lambda x: pd.cut(x['yaw'], bins = [-15, -5, 5, 15], labels = ['left', 'middle', 'right']),
    # human_info['yaw'].apply(lambda x: 'left' if x < -5 else ('middle' if -5 <= x <= 5 else 'right'))
    yaw_direction = lambda x: pd.cut(x['yaw'], bins = [-15, -7, -4, 4, 7, 15], labels = ['left', 'vacant_left', 'middle', 'vacant_right', 'right']),
    # if you want to round to nearest 0.5 instead, use np.round(x['skin_tone'] * 2) / 2
    # ceil rounds up, which is what we want since values include 0.74854187 and there is no 0 for this. Similarly, the highest is 5.66467382, which will round to 6 (highest)
    skin_tone_rounded = lambda x: np.ceil(x['skin_tone']),
    # for this, just attempt light = 1 and 2, dark = 4, 5, 6
    skin_labels = lambda x: np.where(x['skin_tone_rounded'] <= 2., 'light', np.where(x['skin_tone_rounded'] >= 4., 'dark', 'misc')),
    full_image_path = lambda x: x.apply(lambda y: get_full_image_path(folder = 'data/diverse_human_faces', human_id = int(y['human_id']), render_id = y['render_id']), axis = 1),
    image_path = lambda x: x.apply(lambda y: get_image_path(human_id = int(y['human_id']), render_id = y['render_id']), axis = 1),
)

# save csv and parquet
human_info.to_csv('../data/human_info.csv', index = False)
pq.write_table(pa.Table.from_pandas(human_info), '../data/human_info.parquet')
```

The current layout of `data/diverse_human_faces/` has 70,000 files. Adding folders for the `human_id` to meet the requirements for `torchvision`. 


```python
df = pq.read_table('../data/human_info.parquet').to_pandas()

CURRENT_DIR = os.getcwd().split('/')[-1]
PATH = '../data/diverse_human_faces'
FOLDER_VALUES_N = np.vectorize(lambda x: int(re.search(r'^(\d+)', x).group(1)) if re.search(r'^(\d+)', x) else -1)

# sorting the folders by the number in the folder name 
DIR_FILES = np.array([f for f in os.listdir(PATH) if f != '.DS_Store' and f != 'metadata.jsonl'])
DIR_FILES = DIR_FILES[np.argsort(FOLDER_VALUES_N(DIR_FILES))]

CURRENT_PATH = '../data/diverse_human_faces'
NEW_PATH = '../data/faces_sorted'

# create columns for the filenames of the images and jsons
df = df[['human_id', 'render_id']].assign(
    image_name = lambda x: x['render_id'].astype(str) + '.cam_default.f_1.rgb.png',
    json_name = lambda x: x['render_id'].astype(str) + '.cam_default.f_1.info.json'
)

for i, row in df.iterrows():
    human_id = str(row['human_id'])
    render_id = str(row['render_id'])
    image_name = row['image_name']
    json_name = row['json_name']
    
    # create the folder for each human_id
    if not os.path.exists(os.path.join(NEW_PATH, human_id)): 
        # e.g. # data/faces_sorted/325
        os.makedirs(os.path.join(NEW_PATH, human_id), exist_ok = True)
    
    # move the files to the corresponding human_id folder
    # from      data/diverse_human_faces/0.cam_default.f_1.rgb.png
    # to        data/faces_sorted/325/0.cam_default.f_1.rgb.png  
    os.rename(os.path.join(CURRENT_PATH, image_name), os.path.join(NEW_PATH, human_id, image_name)) 
    os.rename(os.path.join(CURRENT_PATH, json_name), os.path.join(NEW_PATH, human_id, json_name))

# checking that it worked as planned. each human_id folder should have 100 images and 100 jsons (200 files total)

count = 0

for folder in os.listdir(NEW_PATH):
    folder_path = os.path.join(NEW_PATH, folder)
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        if len(files) != 200:
            print(f"Folder '{folder}' does not have 200 files. It has {len(files)} files.")
        else:
            count += 1

print(f"{count} folders have 200 files.")
```

Removing `human_id` where the `skin_tone` == 3 


```python
df = pq.read_table('../data/human_info.parquet').to_pandas()
# now 5422 rows
df = df.query('skin_labels != "misc" & yaw_direction != "vacant"').reset_index(drop = True).assign(
    image_path = lambda x: '../data/' + x['image_path']
)

CURRENT_PATH = '../data/diverse_human_faces'
NEW_PATH = '../data/faces_sorted'

# create human_id folders in new directory
for idx in df['human_id'].unique():
    os.makedirs(os.path.join(NEW_PATH, str(idx)), exist_ok = True)

for index, row in tqdm(df.iterrows(), total = len(df)):

    # image_path             ../data/diverse_human_faces/325/0.cam_default.f_1.rgb.png
    image_path = row['image_path']
    # image_dest             ../data/faces_sorted/325/0.cam_default.f_1.rgb.png
    image_dest = image_path.replace(CURRENT_PATH, NEW_PATH)
    shutil.copy2(image_path, image_dest)

df.to_csv('../data/full_data.csv', index = False)
```


```python
# checking that the count from df matches the directories 
files_list = []

for folder in os.listdir(NEW_PATH):
    folder_path = os.path.join(NEW_PATH, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            files_list.append(os.path.join(folder_path, file))

print(f"Found {len(files_list)} images in {len(os.listdir(NEW_PATH))} human_id.")
```

Creating aggregated datasets

Proportion of `skin_tone` is the same for both groups, but only use images looking straight for the undersampled group. Hhere you are using straight as the only position for the undersampled group


```python
df = pq.read_table('../data/human_info.parquet').to_pandas()

df_summary = (df
    .groupby('human_id')
    .size()
    .reset_index()
    .rename(columns = {0: 'n'})
    .merge(
        df.drop_duplicates(subset = ['human_id'])[['human_id', 'skin_labels']].reset_index(drop = True)
))

df_summary.sort_values('n', ascending = False)
```

Creating the biased datasets.


```python
dark_undersampled_df = pd.concat([
    # all values where skin_labels == "dark" and yaw_direction == "middle" (785, 14)
    # 473 male, 312 female
    df.query('skin_labels == "dark" and yaw_direction == "middle"'),
    # sample light skin faces, across the three possible yaw_directions (1170, 14)
    # 450 male, 720 female
    df.query('skin_labels == "light"').groupby('human_id').sample(n = 30, replace = False, random_state = RANDOM_SEED)
]).reset_index(drop = True)

dark_undersampled_df.to_csv('data/dark_undersampled.csv', index = False)

light_undersampled_df = pd.concat([
    # all values where skin_labels == "dark" and yaw_direction == "middle" (1012, 14)
    # 620 male, 393 female
    df.query('skin_labels == "light" and yaw_direction == "middle"'),
    # sample light skin faces, across the three possible yaw_directions (870, 14)
    # 540 male, 330 female
    df.query('skin_labels == "dark"').groupby('human_id').sample(n = 30, replace = False, random_state = RANDOM_SEED)
]).reset_index(drop = True)

light_undersampled_df.to_csv('data/light_undersampled.csv', index = False)
```


```python
!mkdir ../data/dark_undersampled ../data/light_undersampled
```

Create the Dark Undersampled Dataset


```python
dark_undersampled_df = pd.read_csv('data/dark_undersampled.csv')

dark_undersampled_df = dark_undersampled_df.assign(
    image_path_full = lambda x: 'data/diverse_human_faces/' + x['human_id'].astype(str) + '/' + x['render_id'].astype(str) + '.cam_default.f_1.rgb.png',
    json_path_full = lambda x: 'data/diverse_human_faces/' + x['human_id'].astype(str) + '/' + x['render_id'].astype(str) + '.cam_default.f_1.info.json'
)

CURRENT_PATH = "data/diverse_human_faces"
DARK_PATH = "data/dark_undersampled"

for index, row in dark_undersampled_df.iterrows():

    # create human_id folders in dark_undersampled
    for idx in dark_undersampled_df['human_id'].unique():
        os.makedirs(os.path.join(DARK_PATH, str(idx)), exist_ok = True)

    # image_path             data/diverse_human_faces/110/101.cam_default.f_1.rgb.png
    image_path = row['image_path_full']
    # image_dest           data/dark_undersampled/110/101.cam_default.f_1.rgb.png   
    image_dest = image_path.replace(CURRENT_PATH, DARK_PATH)
    # image_dest.split('/')[-2], e.g. 110
    shutil.copy2(image_path, image_dest)

    # json_path             data/diverse_human_faces/110/101.cam_default.f_1.info.json
    json_path = row['json_path_full']
    # json_dest           data/dark_undersampled/110/101.cam_default.f_1.info.json
    json_dest = json_path.replace(CURRENT_PATH, DARK_PATH)
    shutil.copy2(json_path, json_dest)
```

Creating the light undersampled dataset.


```python
light_undersampled_df = pd.read_csv('data/light_undersampled.csv')

light_undersampled_df = light_undersampled_df.assign(
    image_path_full = lambda x: 'data/diverse_human_faces/' + x['human_id'].astype(str) + '/' + x['render_id'].astype(str) + '.cam_default.f_1.rgb.png',
    json_path_full = lambda x: 'data/diverse_human_faces/' + x['human_id'].astype(str) + '/' + x['render_id'].astype(str) + '.cam_default.f_1.info.json'
)

CURRENT_PATH = "data/diverse_human_faces"
LIGHT_PATH = "data/light_undersampled"

for index, row in tqdm(light_undersampled_df.iterrows(), total = len(light_undersampled_df)):

    # create human_id folders in light_undersampled
    for idx in light_undersampled_df['human_id'].unique():
        os.makedirs(os.path.join(LIGHT_PATH, str(idx)), exist_ok = True)

    # images
    image_path = row['image_path_full']
    image_dest = image_path.replace(CURRENT_PATH, LIGHT_PATH)
    shutil.copy2(image_path, image_dest)

    # jsons
    json_path = row['json_path_full']
    json_dest = json_path.replace(CURRENT_PATH, LIGHT_PATH)
    shutil.copy2(json_path, json_dest) 
```

If you want to evaluate the numbers of each dataset:

```python
dark_undersampled = pd.read_csv('../data/dark_undersampled.csv')
light_undersampled = pd.read_csv('../data/light_undersampled.csv')

counts_dark = dark_undersampled.assign(gender_label = lambda x: np.where(x['sex'] == .0, 'Male', 'Female')).value_counts('gender_label').reset_index(name='count')
total_count = counts_dark['count'].sum()
counts_dark['proportion'] = counts_dark['count'] / total_count
counts_dark.loc['Total'] = ['', total_count, '']

counts_light = light_undersampled.assign(gender_label = lambda x: np.where(x['sex'] == .0, 'Male', 'Female')).value_counts('gender_label').reset_index(name='count')
total_count_light = counts_light['count'].sum()
counts_light['proportion'] = counts_light['count'] / total_count_light
counts_light.loc['Total'] = ['', total_count_light, '']

comp = pd.concat([
    counts_light.set_index('gender_label').rename(columns={'count': 'light_count', 'proportion': 'light_female_proportion'}), 
    counts_dark.set_index('gender_label').rename(columns={'count': 'dark_count', 'proportion': 'dark_female_proportion'})
    ], axis=1, sort=False)

comp.rename({'': 'Total'})

comp
```
