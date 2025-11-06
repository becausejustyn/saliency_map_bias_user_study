# Supplementary Material for Do Explanations Expose Bias? How Saliency Maps Affect Judgements of Biased Face-Recognition Models

The dataset used for training the models is based on the [Diverse Human Faces Dataset](https://synthesis.ai/diverse-human-faces-dataset/). Follow the instructions on their website to download the dataset. The dataset is not included in this repository due to its size. 

> 📝 **Note:** When you download this dataset, it will download all 70,001 files into your active directory, so make sure you create a directory first.

## Notebooks

1. `01_Dataset_Prep.ipynb`: Prepares the dataset by creating labels and organising files.
2. `02_Preprocessing_Data.ipynb`: Preprocesses the data for training the models.
3. `03_Training.ipynb`: Contains the training code for the models.
4. `04_Explainations.ipynb`: Generates explanations for the trained models.
5. `05_Model_Evaluation.ipynb`: Evaluates the models' performance.
6. `06_Preparing_Trials.ipynb`: Prepares trials for the user study.
7. `07_Analysis.ipynb`: Data analysis of the user study results.



`src`: Containers helper scripts for the project.

## Supplementary Data Stored Over at [Zenodo](https://zenodo.org/records/16907686)

### Models

- `fair.pt`: the balanced model.
- `light_undersampled.pt`: the light skin tone undersampled model.
- `dark_undersampled.pt`: the dark skin tone undersampled model.

### Datasets

These are the subset of each dataset used to train the models above.

- `light_undersampled_cropped.zip`: the dataset for the `light_undersampled.pt`.
- `dark_undersampled_cropped.zip`	: the dataset for the `dark_undersampled.pt`.
- `diverse_human_faces_cropped.zip`: the dataset for the `fair.pt`.

### Trial Data

These relate to the data used during the trials (experiement ran on Profilic).

- `trial_dataset.csv`: overview of the images used for each trial. `model_1` and `model_2` refers to which models relate to the specific trial. 
- `trial_dataset_long.csv`: a deeper overview of `trial_dataset.csv`, giving specific details about each image.
- `trial_images.zip`: for each trial n, all the images that were generated. Refer to `trial_dataset_long.csv` to see which images were used in each trial. 

### Metadata

This is the metadata from the original dataset, [Diverse Human Faces Dataset](https://synthesis.ai/diverse-human-faces-dataset/).

- `dataset_metadata.csv`: metadata for all the images in original dataset with labels created in `01_Dataset_Prep.ipynb`.



## To do

- [ ] Include source for for visualisations.
- [ ] Update `07_Analysis_Prep.ipynb` to include `binom.test` for the user study results. Also check which functions included are still needed.
- [ ] Update `04_Explainations.ipynb` to include the source code used to generate the explanations.
- [ ] Update `05_Model_Evaluation.ipynb` to include the source code used to evaluate the models.
