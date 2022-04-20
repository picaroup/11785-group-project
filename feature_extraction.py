import librosa
import os
import pandas as pd
import numpy as np

configs = [
    [2048, 512, 2048],
    [2048, 512, 1024],
    [2048, 1024, 1024],
    [1024, 512, 1024],
    [1024, 256, 1024]
]

def process_data(personality_dir="data/Personality_Scores", metadata_dir="data/Metadata", audio_dir="data/Audio_clips"):
    df_personality = get_personality_scores(personality_dir)
    df_metadata = get_metadata(metadata_dir)
    df_feature = get_features(audio_dir)
    
    df = pd.merge(df_personality, df_metadata, left_on="Clip_ID", right_on="Clip_ID")
    df = pd.merge(df, df_feature, left_on="Clip_ID", right_on="Clip_ID")
    
    return df

def get_personality_scores(data_dir_path):
    df = pd.read_csv(data_dir_path + "/Score_011.csv")

    for i in range(1, 11):
        df_tmp = pd.read_csv(data_dir_path + f"/Score_0{i:02d}.csv")
        
        df["Extraversion"] = df["Extraversion"] + df_tmp["Extraversion"]
        df["Agreeableness"] = df["Agreeableness"] + df_tmp["Agreeableness"]
        df["Conscientiousness"] = df["Conscientiousness"] + df_tmp["Conscientiousness"]
        df["Neuroticism"] = df["Neuroticism"] + df_tmp["Neuroticism"]
        df["Openness"] = df["Openness"] + df_tmp["Openness"]

    df["Extraversion"] = df["Extraversion"] / 11
    df["Agreeableness"] = df["Agreeableness"] / 11
    df["Conscientiousness"] = df["Conscientiousness"] / 11
    df["Neuroticism"] = df["Neuroticism"] / 11
    df["Openness"] = df["Openness"] / 11
    
    return df

def get_metadata(data_dir_path):
    df = pd.read_csv(data_dir_path + "/Metadata.csv")
    return df

def get_features(data_dir_path):
    data = []
    
    for filename in os.listdir(data_dir_path):
        f = os.path.join(data_dir_path, filename)
        
        y, sr = librosa.load(f)
        output = []
        pad_len = 0
        for config in configs:
            s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=config[0], hop_length=config[1], win_length=config[2])
            pad_len = max(pad_len, len(s[0]))
            output.append(s)
        
        for i, cur in enumerate(output):
            cur = np.pad(cur, ((0,0), (0, pad_len-len(cur[0]))))
            output[i] = cur
        output = np.array(output)

        data.append([filename.split(".")[0], output])
    
    df = pd.DataFrame(data, columns=["Clip_ID", "features"])
    return df

def main():
    df = process_data()
    df.to_pickle("data/processed_data.pkl")

if __name__ == "__main__":
    main()

# https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
# play with Parameters
# multiple features. 2d -> 3d