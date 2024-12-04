import os 
import librosa
import numpy as np 

data_dir = "/path/to/common-voice-data"

save_dir = "/path/to/preprocessed-data"

sampling_rate = 16000
duration = 5

for subdir in os.listdir(data_dir):
    subpath = os.path.join(data_dir, subdir)
    if not os.path.isdir(subpath):
        continue 
    for filename in os.listdir(subpath):
        if not filename.endswith(".mp3"):
            continue
        filepath = os.path.join(subpath, filename)
        # Load the audio file 
        y, sr = librosa.load(filepath, sr = sampling_rate, duration=duration)
        # Save the preprocessed audio file 
        save_path = os.path.join(save_dir, subdir, filename[:-4] + ".npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, y)