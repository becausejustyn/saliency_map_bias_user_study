# saliency_map_bias_user_study

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



To do

- [ ] Include source for for visualisations.
- [ ] Update `07_Analysis_Prep.ipynb` to include `binom.test` for the user study results. Also check which functions included are still needed.
- [ ] Update `04_Explainations.ipynb` to include the source code used to generate the explanations.
- [ ] Update `05_Model_Evaluation.ipynb` to include the source code used to evaluate the models.