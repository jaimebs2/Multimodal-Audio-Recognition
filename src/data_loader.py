"""RAVDESS Data Loader."""
import os
import librosa
import scipy.signal
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class RAVDESSAudioDataLoader:
    """Data loader class for RAVDESS audio dataset."""
    
    def __init__(self, processor, config):
        """
        Initialize the data loader.

        Args:
            data_dir (str): Path to the root folder containing RAVDESS audio data.
            processor: Audio processing function to be applied to the data.
            batch_size (int): Batch size for data loading.
            sampling_rate (int, optional): Sampling rate for audio data. Default is 16000.
            random_state (int, optional): Random seed for data splitting. Default is 42.
        """
        self.data_dir = config["data_dir"]
        self.processor = processor
        self.batch_size = config["model_config"]['batch_size']
        self.val_size = config["model_config"]['val_size']
        self.test_size = config["model_config"]['test_size']
        self.sampling_rate = config["model_config"]['sampling_rate']
        self.random_state = config["model_config"]['seed']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self._load_data()

    def _load_data(self):
        # Load file paths and labels
        file_paths, labels = self._load_data_paths_and_labels()

        # Split data into train, validation, and test sets
        X_train, X_val, y_train, y_val = train_test_split(file_paths, labels, test_size=(self.val_size + self.test_size), random_state=self.random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=1/(10*(self.val_size + self.test_size)), random_state=self.random_state)

        # Create data loaders
        train_dataset = AudioDataset(X_train, y_train, processor=self.processor, sampling_rate=self.sampling_rate)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size)

        val_dataset = AudioDataset(X_val, y_val, processor=self.processor, sampling_rate=self.sampling_rate)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        test_dataset = AudioDataset(X_test, y_test, processor=self.processor, sampling_rate=self.sampling_rate)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

    def _load_data_paths_and_labels(self):
        # Load file paths and corresponding labels from the data directory
        file_paths = []
        labels = []

        actors_folders = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))]
        for actor_folder in actors_folders:
            actor_path = os.path.join(self.data_dir, actor_folder)
            wav_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]
            for wav_file in wav_files:
                file_name = os.path.splitext(wav_file)[0]
                label = file_name.split('-')[2]  # Extract the third number
                file_paths.append(os.path.join(actor_path, wav_file))
                labels.append(label)

        return file_paths, labels

data = RAVDESSAudioDataLoader()


class AudioDataset(Dataset):
    """Custom dataset class for audio data."""

    def __init__(self, file_paths, labels, processor, sampling_rate=16000):
        """
        Initialize the audio dataset.

        Args:
            file_paths (list): List of audio file paths.
            labels (list): List of corresponding labels.
            processor: Audio processing function to be applied to the data.
            sampling_rate (int, optional): Sampling rate for audio data. Default is 16000.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sp_prompt = "<|startofanalysis|><|en|><|emotion|><|en|><|notimestamps|>"

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        #audio, sr = librosa.load(self.file_paths[idx])
        label = int(self.labels[idx])
        #resampled_signal = scipy.signal.resample(audio, int(len(audio) * self.sampling_rate / sr))
        self.query = f"<audio>{self.file_paths[idx]}</audio>{self.sp_prompt}"
        self.audio_info = self.processor.process_audio(self.query)
        inputs = self.processor(self.query, return_tensors='pt', audio_info=self.audio_info)
        inputs = inputs.to(self.device)
        label_tensor = torch.tensor(label, dtype=torch.long).to(self.device)
        if label_tensor == 8:
            label_tensor = torch.tensor(0, dtype=torch.long).to(self.device)
        return inputs, label_tensor
