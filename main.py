import tensorflow as tf
import numpy as np
import warnings
from PIL import Image
import IPython.display as display
import random
from numpy.random import choice
import matplotlib.pyplot as plt, zipfile
import xml.etree.ElementTree as ET
import time
import xml.dom.minidom
from IPython.display import FileLink, FileLinks
import os, glob, logging
from .module1.GAN import make_generator, make_discriminator, smooth_positive_labels, smooth_negative_labels, noisy_labels, discriminator_loss, generator_loss, train_step, train


def run(**kwargs):
    logger = kwargs['logger']
    logger.info("Kernel v1")
    kernel_dir=kwargs['kernel_dir']
    
    #ROOT = kernel_dir
    ROOT = '/Linz2021/input'
    IMAGES = os.listdir(ROOT + '/all/')
    virus_types = os.listdir(ROOT + '/virus_types/')
    
    #Hyperparameters
    dim = 256
    BATCH_SIZE = 16
    noise_dim = 100
    EPOCHS = 20000
    nm=200
    
    logger.info("Start")
    
    # Data Preprocess
    idxIn = 0; namesIn = [] # List of condition
    imagesIn = np.zeros((nm,256,256,3)) # Get Images in A Array

    for virus in virus_types:
        for i in os.listdir(ROOT+'/virus_types/'+virus):
            try: img = Image.open(ROOT+'/all/'+i+'.jpg') 
            except: continue              
            tree = ET.parse(ROOT+'/virus_types/'+virus+'/'+i)
            root = tree.getroot()
            objects = root.findall('object')
            # get images & condition
            for o in objects:
                bndbox = o.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                w = np.min((xmax - xmin, ymax - ymin))
                img2 = img.resize((256,256), Image.ANTIALIAS)
                imagesIn[idxIn,:,:,:] = np.asarray(img2)
                namesIn.append(virus)
                idxIn += 1

    # Normalization
    imagesIn = (imagesIn[:idxIn,:,:,:]-127.5)/127.5
    
    # Change Datatype from float64 to float32 for TF's gradient descent
    imagesIn = tf.cast(imagesIn, 'float32')
    
    logger.info("shape: "+str(imagesIn.shape[0])+str(imagesIn.shape[1])+str(imagesIn.shape[2]))
    logger.info("image load ok")

    
    mirrored_strategy = tf.distribute.MirroredStrategy()
    logger.info("Number of devices: "+str(mirrored_strategy.num_replicas_in_sync))
    
    with mirrored_strategy.scope():
        # implement generator
        generator = make_generator()
        
        # random noise vector
        noise = tf.random.normal([1,noise_dim])
        
        # run the generator model with the noise vector as input
        generated_image = generator(noise, training=False)

        # # implement discriminator
        discriminator = make_discriminator(dim)
    
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
        # optimizers
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)  
    
    # Create Dataset
    ds = tf.data.Dataset.from_tensor_slices(imagesIn).batch(BATCH_SIZE)
    
    # Start Training
    logger.info("Start training")
    train(ds, EPOCHS)  
    logger.info("training ok")
    
    # Model Saving
    filename = '/Linz2021/DGX_Results/gen_model_256_10.h5'
    tf.keras.models.save_model(generator,filename)
    logger.info("save model ok")
    
    # Virus Images Generator
    z = zipfile.PyZipFile('/Linz2021/images_256.zip', mode='w')
    for k in range(1000):
        generated_image = generator(tf.random.normal([1, noise_dim]), training=False)
        f = str(k)+'.png'
        img = ((generated_image[0,:,:,:]+1.)/2.).numpy()
        tf.keras.preprocessing.image.save_img(f,img,scale=True)
        z.write(f); os.remove(f)
    z.close()
    logger.info("generate images ok")
    
    logger.info("End")
