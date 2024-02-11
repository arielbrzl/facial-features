"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.modules import ELU, BatchNorm2d, Conv2d, Flatten, LeakyReLU, Linear, MaxPool2d

# TODO: Choose from either model and uncomment that line
class KeypointModel(nn.Module):
# class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
      """
      Initialize your model from a given dict containing all your hparams
      Warning: Don't change the method declaration (i.e. by adding more
          arguments), otherwise it might not work on the submission server
          
         You could either choose between pytorch or pytorch lightning, 
          by switching the class name line.
      """
      super().__init__()
      self.hparams = hparams
    


      self.func = nn.Sequential(
        # nn.BatchNorm2d(1),
        nn.Conv2d(1, 16, 5), #92
        nn.MaxPool2d(2), #46
        nn.Dropout(),
        nn.Tanh(),
        # nn.BatchNorm2d(16),
        nn.Conv2d(16,32,5), #42
        nn.MaxPool2d(2), #21
        nn.Tanh(),
        # nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, 6), #16
        nn.MaxPool2d(2),
        nn.Dropout(),
        nn.Tanh(),
        # nn.BatchNorm2d(64), 
        nn.Conv2d(64,128, 4), #5
        nn.Dropout(),
        nn.Tanh(),
        # nn.BatchNorm2d(128),
        nn.Flatten(),
        nn.Linear(128*5*5, 1000),
        nn.Tanh(),
        # nn.BatchNorm1d(1000),
        nn.Linear(1000,100),
        nn.Tanh(),
        # nn.BatchNorm1d(100),
        nn.Linear(100,30),
        # nn.BatchNorm1d(30),
      )


    def forward(self, x):
        
      # check dimensions to use show_keypoint_predictions later
      if x.dim() == 3:
          x = torch.unsqueeze(x, 0)

      N, C, H , W = x.shape
      x= self.func(x)
      x=torch.reshape(x, (N, 30))
      return x



class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
