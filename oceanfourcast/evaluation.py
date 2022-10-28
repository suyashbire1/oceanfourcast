import os
import matplotlib.pyplot as plt
import json
from oceanfourcast import load, fourcastnet

class experiment():
    def __init__(self, expt_dir, name):
        self.name = name
        self.expt_dir = expt_dir
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

    def recreate_model(self, epoch=None, device='cpu'):
        self.model = fourcastnet.AFNONet(embed_dim=self.embed_dims,
                                         patch_size=self.patch_size,
                                         sparsity=self.sparsity,
                                         img_size=[self.image_height, self.image_width],
                                         in_channels=self.in_channels,
                                         out_channels=self.out_channels,
                                         affine_batchnorm=self.affine_batchnorm,
                                         drop_rate=self.drop_rate)
        if epoch is None:
            epoch = self.best_vloc_epoch
        model_path = os.path.join(self.expt_dir, f'model_epoch_{epoch}')
        self.model.load_state_dict(torch.load(model_path),
                                   map_location=torch.device(device))
