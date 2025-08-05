# Imports
## Standard
## Local
from .helpers import print_row
from .metrics import MetricManager

## Third-Party
import numpy as np
import torch
import pretty_plotly as pp

# Functions and Classes
class MLPTrainer:
    def __init__(self, type="reg", is_multiclass=False):
        assert type in ["reg", "cls"], "Trainer type must be 'reg' or 'cls'."

        self.type = type
        self.is_multiclass = is_multiclass

        # Set the headers
        self.header = print_row(["TRAINING SET", "VALIDATION SET"], 120)
        if type == "reg":
            self.secondary_header = print_row(
                ["Loss", "RMSE", "MSE", "MAE", "R2"]*2, 
                120
            )
        elif type == "cls":
            self.secondary_header = self.secondary_header = print_row(
                ["Loss", "Accuracy", "Precison", "Recall", "F1"]*2,
                120,
            )
        
        self.metrics = MetricManager(type=type, is_multiclass=is_multiclass)

    def train(
            self, 
            model, 
            num_epochs, 
            optimiser, 
            loss_fn, 
            train_loader, 
            val_loader=None, 
            pred_threshold=0.5,
            device="cpu",
        ):
        print("    " + self.header)
        print("    " + self.secondary_header)

        for e in range(num_epochs):
            # Training and evaluating model on training set data
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []

            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimiser.zero_grad()
                logits = model(X)
                if self.is_multiclass:
                    loss = loss_fn(logits, y.squeeze(1))
                else:
                    loss = loss_fn(logits, y)
                loss.backward()
                optimiser.step()
                train_loss += loss.item() * X.size(0)

                # Get predictions for metrics calculations
                if (self.type == "cls") and (not self.is_multiclass): # preds for binary classification
                    preds = (logits > pred_threshold).float()
                elif (self.type == "cls") and self.is_multiclass: # preds for multiclass classification
                    _, preds = torch.max(logits, 1)
                elif self.type == "reg": # preds for regression
                    preds = logits.float()
                
                train_preds.extend(preds.detach().cpu().numpy())
                train_labels.extend(y.cpu().numpy())

            train_preds = np.array(train_preds).flatten()
            train_labels = np.array(train_labels).flatten()

            # Computing training loss and other metrics
            train_loss = train_loss / len(train_loader)
            self.metrics.calculate(train_loss, train_preds, train_labels, append_list="train")

            # Evaluate on validation set
            if val_loader is not None:
                model.eval()
                val_loss = 0
                val_preds = []
                val_labels = []

                with torch.no_grad():
                    for X, y in val_loader:
                        X, y = X.to(device), y.to(device)
                        logits = model(X)
                        if self.is_multiclass:
                            loss = loss_fn(logits, y.squeeze(1))
                        else:
                            loss = loss_fn(logits, y)
                        val_loss += loss.item()

                        # Get predictions for metrics calculations
                        if (self.type == "cls") and (not self.is_multiclass): # preds for binary classification
                            preds = (logits > pred_threshold).float()
                        elif (self.type == "cls") and self.is_multiclass: # preds for multiclass classification
                            _, preds = torch.max(logits, 1)
                        elif self.type == "reg": # preds for regression
                            preds = logits.float()
                        
                        val_preds.extend(preds.detach().cpu().numpy())
                        val_labels.extend(y.cpu().numpy())

                val_preds = np.array(val_preds).flatten()
                val_labels = np.array(val_labels).flatten()

                # Computing validation loss and other metrics
                val_loss = val_loss / len(val_loader)
                self.metrics.calculate(val_loss, val_preds, val_labels, append_list="val")

            # Print results for current epoch
            result_row_str = self.print_metrics(e)
            print(result_row_str)