import os
import uuid
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from models.MobileNet.mobilenet_architecture import MobileNet, generate_mlp_architecture
from utils.training_utils import train_model
from utils.evaluate_utils import evaluate_model


def warm_up_mobilenet():
    pass

def specialized_training_mobilenet():
    pass