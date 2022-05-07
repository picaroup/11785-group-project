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
The most up-to-date model is `pipeline_colab_version.ipynb`. This code will create a pandas dataframe of the features and labels, create the data loaders, and run the models

