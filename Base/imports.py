import gym
import os
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import gc
import logging as log
import tempfile
import glob
from gym import wrappers
#get gifs function
from IPython.display import HTML
import subprocess
import io
import base64
import json



log.basicConfig(format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p', filename='myapp.log', level=log.INFO)

log.info('Started')