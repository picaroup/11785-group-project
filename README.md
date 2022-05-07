# CNNs and LSTMs: Personality Prediction From Voice
## 11-785 Final Project

## Data
### Download Data
- To download the data, in the /data/ directory, run `bash download.sh`

### Numerical Feature Extraction
- In the /Code/ directory, run the `python3 feature_extraction.py`
- The process data will be stored in /data/processed_data.pkl
- The data is the average judge score for each of the audio files
- This feature extraction method is used in our analysis

### Categorical Feature Extraction
- In the /Code/ directory, run the `categorical_feature_extraction.ipynb` file
- This creates a score of 0 or 1 for each personality trait and each audio file
- For each judge, if the score for that presonality trait is more than the judge's average score fo the perosnality trait, it is assigned a score of 1 for that judge. If it is below the average, it is assigned a score of 0 for that judge
- If the audio file is assigned a score of 1 from 6 or more of the 11 judges for a personality trait, it is assigned an overall score of 1 for that personality trait
- If it is not assigned a score of 1 from 6 or more judges for a personality trait, it is assigned an overall score of 0 for that personality trait

## Models
- The models are in the `/Code/` directory
- The most up-to-date model is `pipeline_colab_version.ipynb`. This code will create a pandas dataframe of the features and labels, create the data loaders, and run the models
- The `pipeline(lstm).ipynb` file creates the model with the CNN feature extraction layer and the LSTM layer
  - Guassian Noise generation is included in the data loader for the training data
  - Parameters used for Ablation study can be specified before creating the model
  - The model parameters and training and validation accuracies are saved to a csv file on the Google Drive
  - Graphs of the training and validation data for each of the personality traits are generated

- The `baseline_all5_feats_final.ipynb` file makes redictions for the five personality traits based on a logistic regression model

- The `CNN_complex_N_blocks_with_plots.ipynb` files creates CNN models with blocks containing spatial and feature mixing CNN layers with `N` blocks in a style inspired by `Convnext`


