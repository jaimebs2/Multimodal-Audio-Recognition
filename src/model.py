"""Model Loader."""
import torch
from transformers import WhisperProcessor, WhisperForAudioClassification

class ModelLoader:
    """
    ModelLoader class for loading and configuring a Whisper model for audio classification.

    Args:
        config (Namespace): Configuration containing model, processor, and freezing layer information.
    """

    def __init__(self, config):
        """
        Initializes the ModelLoader and loads the model and processor.

        Args:
            config (Namespace): Configuration containing model, processor, and freezing layer information.

        Returns:
            model (WhisperForAudioClassification): Loaded Whisper model for audio classification.
            processor (WhisperProcessor): Loaded Whisper processor for data preprocessing.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.classifier, self.processor = self.load()

    def freeze_layers(self):
        """
        Freezes specified layers of the loaded Whisper model.

        Args:
            config (Namespace): Configuration containing freezing layer information.
        """
        if 'classifier' in self.config['frozen_layers']:
            for param in self.classifier.encoder.parameters():
                param.requires_grad = False
        if 'projector' in self.config['frozen_layers']:
            for param in self.classifier.projector.parameters():
                param.requires_grad = False
        self.classifier = self.classifier.to(self.device)

    def load(self):
        """
        Loads the Whisper model and processor based on the provided configuration.

        Args:
            config (Namespace): Configuration containing model, processor, and output size information.

        Returns:
            model (WhisperForAudioClassification): Loaded Whisper model for audio classification.
            processor (WhisperProcessor): Loaded Whisper processor for data preprocessing.
        """
        self.processor = WhisperProcessor.from_pretrained(self.config['processor'], device=self.device)
        self.classifier = WhisperForAudioClassification.from_pretrained(self.config['model'], num_labels=self.config['output_size'])
        self.freeze_layers()
        return self.classifier, self.processor
