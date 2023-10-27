import wandb
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from config import config
from src.data_loader import RAVDESSAudioDataLoader
from src.model import ModelLoader
from src.trainer import AudioModelTrainer

epochs = [50, 100, 150]
lrs = [1e-2, 1e-3, 1e-4]
batch_sizes = [8, 16, 32]
frozen_layers = [['classifier', 'projector'], ['classifier']]

if __name__=="__main__":
    name = config['model_config']['model'].split("/")[-1] +'_'+ wandb.util.generate_id()

    wandb.init(
        entity=config['entity'],
        project=config["project"],
        config=config["model_config"],
        name=name
    )

    model = ModelLoader(config = config["model_config"])
    ravdess_loader = RAVDESSAudioDataLoader(model.processor, config)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=config["model_config"]["learning_rate"])
    trainer = AudioModelTrainer(model = model.classifier, data_loader = ravdess_loader, criterion = criterion, optimizer = optimizer, config=config, wandb_run=wandb)
    trainer.train()
