"""Model Loader."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        self.model, self.tokenizer = self.load()

    def freeze_layers(self):
        """
        Freezes specified layers of the loaded Whisper model.

        Args:
            config (Namespace): Configuration containing freezing layer information.
        """
        if 'lm_head' in self.config['frozen_layers']:
            for param in self.model.lm_head.parameters():
                param.requires_grad = False
        if 'wte' in self.config['frozen_layers']:
            for param in self.model.wte.parameters():
                param.requires_grad = False
        if 'audio_encoder' in self.config['frozen_layers']:
            for param in self.model.audio.parameters():
                param.requires_grad = False
        if 'llm' in self.config['frozen_layers']:
            for param in self.model.h.parameters():
                param.requires_grad = False

    def load(self):
        """
        Loads the Whisper model and processor based on the provided configuration.

        Args:
            config (Namespace): Configuration containing model, processor, and output size information.

        Returns:
            model (WhisperForAudioClassification): Loaded Whisper model for audio classification.
            processor (WhisperProcessor): Loaded Whisper processor for data preprocessing.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.config['processor'], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.config['processor'],trust_remote_code=True).to(self.device)

        return model, tokenizer
