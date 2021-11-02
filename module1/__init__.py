import numpy as np
import random
from numpy.random import choice
import matplotlib.pyplot as plt, zipfile
import os, glob, logging

def run(**kwargs):
    logger = kwargs['logger']
    logger.info("Kernel v1")
    kernel_dir=kwargs['kernel_dir']
    
    #ROOT = kernel_dir
    ROOT = '/mnt/mldp-cifs-nas/YFYANGD/JupyterWorkspace/Linz2021/input'
    IMAGES = os.listdir(ROOT + '/all/')
    virus_types = os.listdir(ROOT + '/virus_types/')
    
    #Hyperparameters
    dim = 256
    BATCH_SIZE = 16
    noise_dim = 100
    EPOCHS = 20000
    nm=200
