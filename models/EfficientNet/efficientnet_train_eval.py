import os
import uuid
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from models.EfficientNet.efficientnet_architecture import EfficientNet, generate_efficientnet_architecture
from utils.training_utils import train_model
from utils.evaluate_utils import evaluate_model


def warm_up_efficientnet():
    pass

def specialized_training_efficientnet():
    pass