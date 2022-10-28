import os
import matplotlib.pyplot as plt
import json

class experiment():
    def __init__(self, expt_dir, name):
        self.name = name
        with open(os.path.join(expt_dir, "logfile.json"), 'r') as f:
            logs = json.load(f)
            for k, v in logs.items():
                setattr(self, k, v)

    def plot_train_loss(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(self.training_loss, label=self.name)
        ax.set_xlabel('Minibatch')
        ax.set_ylabel('Train Loss')
        return ax.get_figure()

    def plot_train_valid_loss(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(self.avg_training_loss, label=self.name + ' train loss')
        ax.plot(self.validation_loss, label=self.name + ' valid loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.grid()
        return ax.get_figure()
