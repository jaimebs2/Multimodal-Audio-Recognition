"""Model and Run Configuration."""

config = {
    "entity": "jaime-bellver",
    "data_dir": "/home/jbellver/datasets/RAVDESS-Audio",
    "project": "Multimodal-Audio-Recognition",
    "model_config": {
        "processor": "openai/whisper-large-v2",
        "model": "openai/whisper-large-v2",
        "dataset": "RAVDESS",
        "output_size": 8,
        "sampling_rate": 16000,
        "epochs": 100,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "frozen_layers": ['classifier', 'projector'],
        "train_size": 0.8,
        "val_size": 0.2,
        "test_size": 0.1,
        "seed": 42,
        "print_interval":9
    }
}
