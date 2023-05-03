import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
import sys
import yaml
from tensorflow.keras.utils import Sequence 
from violinmatch.utils.config import make_secrets
from violinmatch.utils.drive import retrieve_img
from violinmatch.utils.database import connect_db
from violinmatch.recognition.input_generator import load_violin_cropped, get_id_keypoints
from violinmatch.recognition.metrics import pairwise_distances
from tqdm import tqdm
import time




def make_ID_mean_embeddings(engine, secrets, df, model, size : int = 224, square : bool = False):
    ID_list = df.ID.unique()
    # initialize embedding list
    embeddings = tf.expand_dims(tf.zeros(model.layers[-1].output_shape[1:]), 0)
    for ID in tqdm(ID_list):
        img_ids = df.loc[df.ID == ID,"photo_id"].unique()
        imgs = load_violin_cropped(img_ids, df.loc[df.photo_id.isin(img_ids), :], engine, secrets, size, square, False)
        imgs = tf.expand_dims(tf.reduce_mean(model(imgs), 0), 0)
        embeddings = tf.concat([embeddings, imgs], 0)
    embeddings = embeddings[1:]
    return embeddings


def order_ID_list(ID_list, pairwise_dist):
    """ Function used for offline mining
    Args:
    - ID_list: list of all image IDs used for model training
    - pairwise_dist: pairwise distance matrix of corresponding embedding averaged over IDs. Must be ordered according to ID_list. """

    assert ID_list.shape[0]==tf.shape(pairwise_dist)[0], f"{ID_list.shape[0]}, {tf.shape(pairwise_dist)[0]}"
    pairwise_dist = pairwise_dist.numpy() # transform into an array, as for some unknown reason, tf.argsort gives huge numbers while np.argsort worls just fine
    index = np.array(list(range(ID_list.shape[0])))
    new_ID_list = np.array([])
    pairwise_dist = np.argsort(pairwise_dist)
    for i in tqdm(range(ID_list.shape[0])):
        if i in index:
            new_ID_list = np.append(new_ID_list, ID_list[i])
            index = np.delete(index, 0)
            j = 0
            try:
                while pairwise_dist[i, j] not in index:
                    j+=1
            except: 
                print(pairwise_dist[i])
                print(j)
            index = np.delete(index, np.argwhere(index == pairwise_dist[i, j]))
            new_ID_list = np.append(new_ID_list, ID_list[pairwise_dist[i, j]])
        else: 
            pass

    assert ID_list.shape[0] == new_ID_list.shape[0], f"{ID_list.shape[0]}, {new_ID_list.shape[0]}"   
    return(new_ID_list)


class OfflineMiningInputGenerator(Sequence):
    def __init__(self, df : pd.DataFrame, model, ckpt_path : str, max_epoch : int, batch_size : int = 50, image_size : int = 224 , engine = None, secrets = None, square : bool = True, augmentation : bool = False):
        """ 
        This data generator uses offline mining at the end/beginning of each epoch. 
        The model must be the same as the training model and is updated by loading the weights of the checkpoint generated at the end of each epoch.
        The batch size have to be an even number.
        """
        if not secrets:
            self.secrets = make_secrets()
        else:
            self.secrets = secrets
        if not engine:
            self.engine = connect_db(secrets)
        else : 
            self.engine = engine

        self.model = model

        self.ckpt_path = ckpt_path

        self.square = square
        self.augmentation = augmentation
        self.img_size = image_size
        # shuffle df
        # batch the IDs, then get photo_ids from IDs for each batch
        self.df = df
        self.IDs = self.df.ID.unique() 

        # first offline mining:
        self.max_epoch = max_epoch
        self.epoch = 0
        self.pre_IDs = np.array([])
        self.on_epoch_end()
        
        self.batch_size = batch_size
        self.num_batches = np.ceil(self.IDs.shape[0] / batch_size)
        self.batch_idx = np.array_split(range(self.IDs.shape[0]), self.num_batches)


    def on_epoch_end(self):

        if self.epoch<self.max_epoch:

            print("***** OFFLINE MINING ******")

            if self.epoch<self.max_epoch:

                if self.epoch > 0:
                    self.model.load_weights(self.ckpt_path)
               
                distances = make_ID_mean_embeddings(self.engine, self.secrets, self.df, self.model, self.img_size, self.square)
                distances = pairwise_distances(distances, self.square)
                self.IDs = order_ID_list(self.IDs, distances)
                print("****** offline mining giving same results? ",np.array_equal(self.IDs, self.pre_IDs))
                self.pre_IDs = self.IDs.copy()
            self.epoch += 1


    def __len__(self):
        return len(self.batch_idx)


    def __getitem__(self, idx):

        labels = self.IDs[self.batch_idx[idx]]
        tt = self.df.loc[self.df.ID.isin(labels), ["ID","photo_id"]].groupby(["ID","photo_id"]).count().reset_index()
        img_ids = tt.photo_id.values
        labels = tt.ID.values
        img_list = load_violin_cropped(img_ids, self.df, self.engine, self.secrets, self.img_size, self.square, self.augmentation)
        return img_list, labels
        

if __name__=="__main__":
    sys.exit()
    with open('../config.yaml') as file:
        secrets = yaml.full_load(file)
    engine = connect_db(secrets)
    df = get_id_keypoints(engine, secrets)
    model = make_model(0, 380, False)
    model.load_weights("data/model1/checkpoint3/checkpt0.ckpt")

    start_time = time.time()
    embeddings = make_ID_mean_embeddings(engine, secrets, df, model, 380)
    print(time.time()-start_time)

    start_time = time.time()
    distances = pairwise_distances(embeddings)
    print(time.time()-start_time)

    ID_list = df.ID.unique()
    start_time = time.time()
    new_ID = order_ID_list(ID_list, distances)
    print(time.time()-start_time)
    np.savetxt("data/ID_list/ID_list0.txt")

