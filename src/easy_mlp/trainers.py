# Imports
## Local
from .helpers import print_row
## Standard
## Third-Party
import pretty_plotly as pp

# Functions and Classes
class MLPTrainer:
    def __init__(self, type="reg"):
        assert type in ["reg", "cls"], "Trainer type must be 'reg' or 'cls'."

        self.type = type
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

    def train(self):
        pass