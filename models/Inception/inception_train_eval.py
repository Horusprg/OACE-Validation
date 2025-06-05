import os
import uuid
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from models.Inception.inception_architecture import Inception, generate_inception_architecture
from utils.training_utils import train_model
from utils.evaluate_utils import evaluate_model


def warm_up_inception():
    pass

def specialized_training_inception():
    pass