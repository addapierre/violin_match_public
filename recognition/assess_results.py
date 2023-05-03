import tensorflow as tf
import numpy as np
import pandas as pd 

# from violinmatch.siamese_network.train_loop import make_model
# from violinmatch.siamese_network.metrics import pairwise_distances
# from violinmatch.siamese_network.input_generator import get_id_keypoints, load_violin_cropped, crop_back, resize_pad
from augmentation import grayscale, contrast
from violinmatch.utils.database import connect_db
from violinmatch.utils.drive import retrieve_img

import yaml
import cv2
import sys
import os
from tqdm import tqdm

def display(image):

    cv2.namedWindow('Positive', cv2.WINDOW_NORMAL)

    while True:

        cv2.imshow('Positive', image)

        if (tt := cv2.waitKey(10) & int(0xFF)) != 255:

            if tt == ord('q'):

                cv2.destroyAllWindows()
                sys.exit()
            if tt == ord('n'):
                break

def acc_bad_predictions(labels, embeddings, n_images = 1):
    """
    metrique: vérifie pour chaque image du batch si il y a au moins une image avec le même ID parmis les n embeddings d'images les plus proches en distance euclidienne.
    """
    labels = tf.squeeze(labels)
    pairwise_dist = pairwise_distances(embeddings, squared=False)
    max_dist = tf.reduce_max(pairwise_dist)
    mask = tf.cast(tf.eye(tf.shape(labels)[0]), tf.float32)
    mask *= max_dist
    result = pairwise_dist+mask
    result = tf.argsort(tf.cast(result, tf.float64), axis = 1, direction="ASCENDING")[:,:n_images] #le cast en tf.float64 est nécessaire pour éviter un bug spécifique au graph mode sur la puce m1.
    result = tf.map_fn(lambda x: tf.gather(params = tf.squeeze(labels), indices=x), result, fn_output_signature = tf.int64)
    result = tf.transpose(result, (1,0))
    accuracy = tf.equal(result, labels)
    #accuracy = tf.map_fn(lambda x: tf.equal(labels, x), result, fn_output_signature = tf.bool)
    accuracy = tf.cast(tf.transpose(accuracy, (1,0)), tf.float32)
    accuracy = tf.reduce_max(accuracy, axis = 1)
     
    return accuracy





if __name__ == "__main__":

    iids = np.load('iid_to_test.npy')    
    df_kp = pd.read_csv('data/arcface_df_2/df_kp.csv', index_col=0)

    if os.path.exists('data/bad_iids.csv'):
        bad_iid = pd.read_csv('data/bad_iids.csv', index_col=0)
        bad_iid = bad_iid[['instance_id']]
        last_iid = bad_iid.instance_id.values[-1]
        i = np.where(iids==last_iid)[0].squeeze()+1
    else:
        bad_iid = pd.DataFrame(data = {'instance_id' : [-1]}, dtype = int )
        i=0
    
    while i < iids.shape[0]:
        iid = iids[i]
        print(i,'/',iids.shape[0]-1, " : ", iid)
        path = df_kp.loc[df_kp.instance_id==iid, 'imguri'].values[0]
        img = cv2.imread(path)
        for j in range(4):
            x = df_kp.loc[df_kp.instance_id==iid, 'x'].values[j]
            y = df_kp.loc[df_kp.instance_id==iid, 'y'].values[j]
            cv2.circle(img, (x,y), 10, (0,255,0), -1)
        cv2.namedWindow('new_image', cv2.WINDOW_NORMAL)
        while True:
            cv2.imshow('new_image', img)
            if (tt := cv2.waitKey(10) & int(0xFF)) != 255:
                if tt == ord('n'):
                    i+=1
                    break
                if tt == ord('b'):
                    i-=1
                    break
                if tt == ord('w'):
                    temp = pd.DataFrame([[iid]],columns = ['instance_id'], dtype=int)
                    bad_iid = pd.concat([bad_iid, temp], ignore_index=True, )
                    bad_iid.to_csv('data/bad_iids.csv')
                    i+=1
                    break
                if tt == ord('q'):
                    sys.exit()




    sys.exit()
    


