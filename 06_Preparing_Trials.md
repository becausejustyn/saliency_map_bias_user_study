```python
import os
import random
import numpy as np
import pandas as pd

RANDOM_SEED = 220223

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
```


```python
all_data = pd.read_csv('../data/human_info.csv')
all_data = all_data.assign(
    base_name = all_data['image_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
)
```


```python
# get_example_paths('../xai_samples/coldnhot/XAI1')
def get_example_paths(xai_path):
    '''
        Get the file paths for all the XAI examples in that folder.
        E.g. xai_samples = get_example_paths('../xai_samples/coldnhot/XAI1')
    '''
    models = ['fair_model', 'dark_undersampled', 'light_undersampled']
    file_paths, subfolder_names = [], []

    for subfolder in models:
        subfolder_path = os.path.join(xai_path, subfolder)
        file_names = [fn for fn in os.listdir(subfolder_path) if not fn.startswith('.') and not fn.endswith('.DS_Store')]
        file_paths.extend([os.path.join(subfolder_path, file_name) for file_name in file_names])

        # Add the subfolder name to the list of subfolder names, with the same length as the number of files in the subfolder
        subfolder_names.extend([subfolder] * len(file_names))

    df = pd.DataFrame({'file_path': file_paths, 'model': subfolder_names})
    df = df.assign(
        # image name, e.g. 734.cam_default.f_1.rgb
        # this is to simplify joining with the human info data
        base_name = df['file_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0][4:])
    )
    return df


def get_block_order(N_PARTICIPANTS = 50, random_seed = 42):

    # create a list of conditions by n_participants
    # 25 fair first, 25 biased first
    fair_first = ['fair_model'] * (N_PARTICIPANTS // 2)
    biased_first = ['biased_model'] * (N_PARTICIPANTS // 2)
    condition_order = fair_first + biased_first

    np.random.seed(random_seed)
    random.seed(random_seed)
    random.shuffle(condition_order)

    # if the first model is biased, then the second model is fair and vice versa
    df_conditions = pd.DataFrame({
        # 'Participant ID': ['P{:02d}'.format(i+1) for i in range(N_PARTICIPANTS)]
        'Participant ID': [f'P{i + 1:02d}' for i in range(N_PARTICIPANTS)],       # range(1, N_PARTICIPANTS + 1),
        'first_model': condition_order,
        'second_model': ['fair_model' if x == 'biased_model' else 'biased_model' for x in condition_order]
        })

    # replace the biased model name with the specific model
    biased_models = np.repeat(['light_undersampled', 'dark_undersampled'], len(df_conditions) // 2) 
    np.random.shuffle(biased_models)

    df_conditions['biased_model'] = biased_models
    df_conditions.loc[df_conditions['first_model'] == 'biased_model', 'first_model'] = df_conditions['biased_model']
    df_conditions.loc[df_conditions['second_model'] == 'biased_model', 'second_model'] = df_conditions['biased_model']
    df_conditions.drop(columns=['biased_model'], inplace = True)

    df_conditions.set_index('Participant ID', inplace = True)
    return df_conditions

def create_participant_matrix(df, n_trials = 32):
    '''Create matrix for each participant'''
    N_PARTICIPANTS = len(df)
    participant_matrix = pd.DataFrame(
        np.zeros((N_PARTICIPANTS, n_trials), dtype = int), 
        columns = [f'Trial {i+1}' for i in range(n_trials)], 
        index = [f'P{i+1:02d}' for i in range(N_PARTICIPANTS)])
    return participant_matrix

def create_sample_paths(participant_df, condition_df, file_dict, random_seed = None):
    '''
        Participant_df: empty matrix for each participant by trial columns
        Condition_df: dataframe with the condition order for each participant
        File_dict: dictionary with the file paths for each model
    '''
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    for participant in participant_df.index:

        first_model = condition_df.loc[participant, 'first_model']
        second_model = condition_df.loc[participant, 'second_model']

        first_model_files = file_dict[first_model]
        second_model_files = file_dict[second_model]

        random.shuffle(first_model_files) 
        random.shuffle(second_model_files) 

        participant_df.loc[participant, 'Trial 1':'Trial 16'] = first_model_files 
        participant_df.loc[participant, 'Trial 17':'Trial 32'] = second_model_files 
    return participant_df
```


```python
# explanation 1 and 2 for the trials
XAI1_PATHS = get_example_paths('../xai_samples/coldnhot/XAI1')
XAI2_PATHS = get_example_paths('../xai_samples/coldnhot/XAI2')

MODELS = XAI1_PATHS['model'].unique().tolist()

file_paths_dict_xai1 = {}
file_paths_dict_xai2 = {}

for model in MODELS:
    file_paths = XAI1_PATHS.query(f'model == "{model}"').reset_index(drop = True)['file_path'].tolist()
    file_paths_dict_xai1[f'{model}'] = file_paths

for model in MODELS:
    file_paths = XAI2_PATHS.query(f'model == "{model}"').reset_index(drop = True)['file_path'].tolist()
    file_paths_dict_xai2[f'{model}'] = file_paths

# file_paths_dict.keys()        dict_keys(['fair_model', 'dark_undersampled', 'light_undersampled'])
# file_paths_dict['fair_model']       file_paths_dict['dark_undersampled']        file_paths_dict['light_undersampled']
```


```python
df_conditions = get_block_order(N_PARTICIPANTS = 50, random_seed = RANDOM_SEED)
part_df = create_participant_matrix(df_conditions)
```


```python
part_paths1 = create_sample_paths(participant_df = part_df, condition_df = df_conditions, file_dict = file_paths_dict_xai1, random_seed = RANDOM_SEED)
part_paths2 = create_sample_paths(participant_df = part_df, condition_df = df_conditions, file_dict = file_paths_dict_xai2, random_seed = RANDOM_SEED)
```


```python
# to get the human_id
merged_df = pd.merge(XAI1_PATHS, all_data[['sex', 'yaw_direction', 'skin_labels', 'base_name', 'human_id']], on = 'base_name', how = 'left')
```


```python
merged_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file_path</th>
      <th>model</th>
      <th>base_name</th>
      <th>sex</th>
      <th>yaw_direction</th>
      <th>skin_labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_5135.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>5135.cam_default.f_1.rgb</td>
      <td>0.0</td>
      <td>middle</td>
      <td>light</td>
    </tr>
    <tr>
      <th>1</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_6736.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>6736.cam_default.f_1.rgb</td>
      <td>1.0</td>
      <td>side</td>
      <td>dark</td>
    </tr>
    <tr>
      <th>2</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_7649.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>7649.cam_default.f_1.rgb</td>
      <td>1.0</td>
      <td>middle</td>
      <td>light</td>
    </tr>
    <tr>
      <th>3</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_6866.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>6866.cam_default.f_1.rgb</td>
      <td>1.0</td>
      <td>middle</td>
      <td>light</td>
    </tr>
    <tr>
      <th>4</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_2728.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>2728.cam_default.f_1.rgb</td>
      <td>1.0</td>
      <td>middle</td>
      <td>light</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>43</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_5248.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>5248.cam_default.f_1.rgb</td>
      <td>0.0</td>
      <td>middle</td>
      <td>light</td>
    </tr>
    <tr>
      <th>44</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_2492.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>2492.cam_default.f_1.rgb</td>
      <td>1.0</td>
      <td>side</td>
      <td>dark</td>
    </tr>
    <tr>
      <th>45</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_9057.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>9057.cam_default.f_1.rgb</td>
      <td>1.0</td>
      <td>side</td>
      <td>dark</td>
    </tr>
    <tr>
      <th>46</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_4839.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>4839.cam_default.f_1.rgb</td>
      <td>0.0</td>
      <td>middle</td>
      <td>light</td>
    </tr>
    <tr>
      <th>47</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_8008.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>8008.cam_default.f_1.rgb</td>
      <td>1.0</td>
      <td>side</td>
      <td>dark</td>
    </tr>
  </tbody>
</table>
<p>48 rows × 6 columns</p>
</div>




```python
# combine the file paths for both XAI1 and XAI2
pd.concat([
    XAI1_PATHS, XAI2_PATHS
], ignore_index = True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file_path</th>
      <th>model</th>
      <th>base_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_5135.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>5135.cam_default.f_1.rgb</td>
    </tr>
    <tr>
      <th>1</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_6736.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>6736.cam_default.f_1.rgb</td>
    </tr>
    <tr>
      <th>2</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_7649.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>7649.cam_default.f_1.rgb</td>
    </tr>
    <tr>
      <th>3</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_6866.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>6866.cam_default.f_1.rgb</td>
    </tr>
    <tr>
      <th>4</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_2728.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>2728.cam_default.f_1.rgb</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>43</th>
      <td>../xai_samples/coldnhot/XAI2/light_undersampled/xai_5248.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>5248.cam_default.f_1.rgb</td>
    </tr>
    <tr>
      <th>44</th>
      <td>../xai_samples/coldnhot/XAI2/light_undersampled/xai_2492.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>2492.cam_default.f_1.rgb</td>
    </tr>
    <tr>
      <th>45</th>
      <td>../xai_samples/coldnhot/XAI2/light_undersampled/xai_9057.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>9057.cam_default.f_1.rgb</td>
    </tr>
    <tr>
      <th>46</th>
      <td>../xai_samples/coldnhot/XAI2/light_undersampled/xai_4839.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>4839.cam_default.f_1.rgb</td>
    </tr>
    <tr>
      <th>47</th>
      <td>../xai_samples/coldnhot/XAI2/light_undersampled/xai_8008.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>8008.cam_default.f_1.rgb</td>
    </tr>
  </tbody>
</table>
<p>96 rows × 3 columns</p>
</div>




```python
# create a df of the file paths for each sample (e.g. input, XAI1, XAI2)
pd.merge(
    XAI1_PATHS.rename(columns = {'file_path': 'file_path1'}),
    XAI2_PATHS.rename(columns = {'file_path': 'file_path2'})[['file_path2', 'base_name']],
    on = 'base_name', how = 'left'
).merge(all_data[['sex', 'yaw_direction', 'skin_labels', 'base_name', 'human_id']], on = 'base_name', how = 'left')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file_path1</th>
      <th>model</th>
      <th>base_name</th>
      <th>file_path2</th>
      <th>sex</th>
      <th>yaw_direction</th>
      <th>skin_labels</th>
      <th>human_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_5135.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>5135.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/fair_model/xai_5135.cam_default.f_1.rgb.png</td>
      <td>0.0</td>
      <td>middle</td>
      <td>light</td>
      <td>288</td>
    </tr>
    <tr>
      <th>1</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_6736.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>6736.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/fair_model/xai_6736.cam_default.f_1.rgb.png</td>
      <td>1.0</td>
      <td>side</td>
      <td>dark</td>
      <td>73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_7649.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>7649.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/fair_model/xai_7649.cam_default.f_1.rgb.png</td>
      <td>1.0</td>
      <td>middle</td>
      <td>light</td>
      <td>35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_6866.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>6866.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/fair_model/xai_6866.cam_default.f_1.rgb.png</td>
      <td>1.0</td>
      <td>middle</td>
      <td>light</td>
      <td>319</td>
    </tr>
    <tr>
      <th>4</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_2728.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>2728.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/fair_model/xai_2728.cam_default.f_1.rgb.png</td>
      <td>1.0</td>
      <td>middle</td>
      <td>light</td>
      <td>63</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>43</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_5248.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>5248.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/light_undersampled/xai_5248.cam_default.f_1.rgb.png</td>
      <td>0.0</td>
      <td>middle</td>
      <td>light</td>
      <td>64</td>
    </tr>
    <tr>
      <th>44</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_2492.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>2492.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/light_undersampled/xai_2492.cam_default.f_1.rgb.png</td>
      <td>1.0</td>
      <td>side</td>
      <td>dark</td>
      <td>107</td>
    </tr>
    <tr>
      <th>45</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_9057.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>9057.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/light_undersampled/xai_9057.cam_default.f_1.rgb.png</td>
      <td>1.0</td>
      <td>side</td>
      <td>dark</td>
      <td>11</td>
    </tr>
    <tr>
      <th>46</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_4839.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>4839.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/light_undersampled/xai_4839.cam_default.f_1.rgb.png</td>
      <td>0.0</td>
      <td>middle</td>
      <td>light</td>
      <td>94</td>
    </tr>
    <tr>
      <th>47</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_8008.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>8008.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/light_undersampled/xai_8008.cam_default.f_1.rgb.png</td>
      <td>1.0</td>
      <td>side</td>
      <td>dark</td>
      <td>89</td>
    </tr>
  </tbody>
</table>
<p>48 rows × 8 columns</p>
</div>




```python
pd.merge(XAI1_PATHS, XAI2_PATHS, on = 'base_name', how = 'left')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file_path_x</th>
      <th>model_x</th>
      <th>base_name</th>
      <th>file_path_y</th>
      <th>model_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_5135.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>5135.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/fair_model/xai_5135.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
    </tr>
    <tr>
      <th>1</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_6736.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>6736.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/fair_model/xai_6736.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
    </tr>
    <tr>
      <th>2</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_7649.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>7649.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/fair_model/xai_7649.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
    </tr>
    <tr>
      <th>3</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_6866.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>6866.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/fair_model/xai_6866.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
    </tr>
    <tr>
      <th>4</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_2728.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>2728.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/fair_model/xai_2728.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>43</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_5248.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>5248.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/light_undersampled/xai_5248.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
    </tr>
    <tr>
      <th>44</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_2492.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>2492.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/light_undersampled/xai_2492.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
    </tr>
    <tr>
      <th>45</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_9057.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>9057.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/light_undersampled/xai_9057.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
    </tr>
    <tr>
      <th>46</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_4839.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>4839.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/light_undersampled/xai_4839.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
    </tr>
    <tr>
      <th>47</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_8008.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>8008.cam_default.f_1.rgb</td>
      <td>../xai_samples/coldnhot/XAI2/light_undersampled/xai_8008.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
    </tr>
  </tbody>
</table>
<p>48 rows × 5 columns</p>
</div>




```python
merged_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file_path</th>
      <th>model</th>
      <th>base_name</th>
      <th>sex</th>
      <th>yaw_direction</th>
      <th>skin_labels</th>
      <th>human_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_5135.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>5135.cam_default.f_1.rgb</td>
      <td>0.0</td>
      <td>middle</td>
      <td>light</td>
      <td>288</td>
    </tr>
    <tr>
      <th>1</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_6736.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>6736.cam_default.f_1.rgb</td>
      <td>1.0</td>
      <td>side</td>
      <td>dark</td>
      <td>73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_7649.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>7649.cam_default.f_1.rgb</td>
      <td>1.0</td>
      <td>middle</td>
      <td>light</td>
      <td>35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_6866.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>6866.cam_default.f_1.rgb</td>
      <td>1.0</td>
      <td>middle</td>
      <td>light</td>
      <td>319</td>
    </tr>
    <tr>
      <th>4</th>
      <td>../xai_samples/coldnhot/XAI1/fair_model/xai_2728.cam_default.f_1.rgb.png</td>
      <td>fair_model</td>
      <td>2728.cam_default.f_1.rgb</td>
      <td>1.0</td>
      <td>middle</td>
      <td>light</td>
      <td>63</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>43</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_5248.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>5248.cam_default.f_1.rgb</td>
      <td>0.0</td>
      <td>middle</td>
      <td>light</td>
      <td>64</td>
    </tr>
    <tr>
      <th>44</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_2492.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>2492.cam_default.f_1.rgb</td>
      <td>1.0</td>
      <td>side</td>
      <td>dark</td>
      <td>107</td>
    </tr>
    <tr>
      <th>45</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_9057.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>9057.cam_default.f_1.rgb</td>
      <td>1.0</td>
      <td>side</td>
      <td>dark</td>
      <td>11</td>
    </tr>
    <tr>
      <th>46</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_4839.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>4839.cam_default.f_1.rgb</td>
      <td>0.0</td>
      <td>middle</td>
      <td>light</td>
      <td>94</td>
    </tr>
    <tr>
      <th>47</th>
      <td>../xai_samples/coldnhot/XAI1/light_undersampled/xai_8008.cam_default.f_1.rgb.png</td>
      <td>light_undersampled</td>
      <td>8008.cam_default.f_1.rgb</td>
      <td>1.0</td>
      <td>side</td>
      <td>dark</td>
      <td>89</td>
    </tr>
  </tbody>
</table>
<p>48 rows × 7 columns</p>
</div>


