"""Trainer."""
import torch
import torch.optim.lr_scheduler as lr_scheduler

class AudioModelTrainer:
    """
    AudioModelTrainer class for training and validating a model on the RAVDESS dataset.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): Loss function for training.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        wandb_run: Weights & Biases run object for logging.
    """

    def __init__(self, model, data_loader, criterion, optimizer, config, wandb_run):
        self.model = model
        self.train_loader = data_loader.train_loader
        self.val_loader = data_loader.val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = config["model_config"]['epochs']
        self.wandb_run = wandb_run
        self.loss_during_training = []
        self.acc_during_training = []
        self.loss_during_valid = []
        self.acc_during_valid = []
        self.steps = config["model_config"]['print_interval']

    def train(self, best_val_loss=float('inf')):
        """
        Train the model for a specified number of epochs.

        Args:
            num_epochs (int): Number of training epochs.
            best_val_loss (float, optional): Best validation loss for early stopping. Default is positive infinity.
        """
        for epoch in range(int(self.epochs)):
            print("\n"+"-"*20+" Epoch {}/{} ".format(epoch+1, int(self.epochs))+"-"*20)

            # Train
            self.model.train()
            print("\nTrain:")
            running_loss = 0.0
            running_corrects = 0 
            for step, batch in enumerate(self.train_loader):
                inputs, labels = batch
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()

                loss = loss.item()
                running_loss += loss

                # Accuracy
                predicted_labels = torch.argmax(outputs.logits, dim=1)
                correct = (predicted_labels == labels).sum().item()
                running_corrects += correct

                samples = len(labels)

                if (step+1)%self.steps==0:
                    avg_loss = running_loss/self.steps
                    avg_acc = running_corrects/(samples*self.steps)
                    self.loss_during_training.append(avg_loss)
                    self.acc_during_training.append(avg_acc)
                    print("\n      Accuracy: {}   |   Loss: {}".format(round(avg_acc,6), round(avg_loss,6)))
                    self.wandb_run.log({"train_loss": avg_loss, "train_acc": avg_acc})
                    running_loss = 0.0
                    running_corrects = 0

            # Validation
            self.model.eval()
            print("\vEvaluate:")
            val_running_loss = 0.0
            val_running_corrects = 0 
            with torch.no_grad():
                for step, val_batch in enumerate(self.val_loader):
                    val_inputs, val_labels = val_batch
                    val_outputs = self.model(val_inputs)
                    val_loss = self.criterion(val_outputs.logits, val_labels)
                    val_loss = val_loss.item()
                    val_running_loss += val_loss

                    # Accuracy
                    val_predicted_labels = torch.argmax(val_outputs.logits, dim=1)
                    val_correct = (val_predicted_labels== val_labels).sum().item()
                    val_running_corrects += val_correct
                    val_samples = len(val_labels)

                    if (step+1)%self.steps==0:
                        avg_loss = val_running_loss / self.steps
                        avg_acc = val_running_corrects / (samples*self.steps)
                        self.loss_during_valid.append(avg_loss)
                        self.acc_during_valid.append(avg_acc)
                        print("\n      Accuracy: {}   |   Loss: {}".format(round(avg_acc,6), round(avg_loss,6)))
                        self.wandb_run.log({"val_loss": avg_loss, "val_acc": avg_acc})
                        val_running_loss = 0.0
                        val_running_corrects = 0 

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_weights = self.model.state_dict()
                        torch.save(best_model_weights, 'model_weights/best_model.pth')
