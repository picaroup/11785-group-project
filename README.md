# CNNs and LSTMs: Personality Prediction From Voice
## 11-785 Final Project

## Data
### Download Data
- To download the data, in the /data/ directory, run `bash download.sh`

### Feature Extraction
- In the /Code/ directory, run the `python3 feature_extraction.py`
- The process data will be stored in /data/processed_data.pkl
- The data is the average judge score for each of the audio files

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


