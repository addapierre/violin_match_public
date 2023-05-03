from gc import callbacks
from operator import truediv
from turtle import clear
from unicodedata import name
import pandas as pd
import numpy as np
import yaml

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4 #EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, layers, losses, metrics

from violinmatch.recognition.input_generator import get_id_keypoints, load_violin_cropped, DataGenerator, DataGeneratorNPair
from violinmatch.recognition.metrics import DistanceAccuracy, FractionPositiveTriplets, ValidationMetricsCallback
from violinmatch.recognition.losses import SemiHardTripletLoss

from violinmatch.utils.config import make_secrets
from violinmatch.utils.database import connect_db




def make_model(fine_tune_layers : int = 20, input_size = 224, square : bool = True, norm : bool = True):
    """
    returns an embedding model using EfficientNet with output size = 256
    efficientnetB0 input size: 224
    efficientnetB4 input size: 380
    """
    if square:
        backbone = EfficientNetB4(include_top=False, input_shape = (input_size, input_size, 3), weights="imagenet")
    else:
        backbone = EfficientNetB4(include_top=False, input_shape = (int(input_size*2), int(input_size/2), 3), weights="imagenet")
    backbone.trainable = False
    if fine_tune_layers:
        for layer in backbone.layers[-fine_tune_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

    #if input size is not a square, use global pooling to avoid conflicting sizes when flattened
    #glob_pool = layers.GlobalAveragePooling2D()(backbone.output)
    flatten = layers.Flatten()(backbone.output) # layers.MaxPooling2D()(backbone.output) #
    flatten = layers.Dropout(0.25)(flatten)
    # dense1 = layers.Dense(512, activation="relu", name="dense1")(flatten)
    # dense1 = layers.BatchNormalization()(dense1)
    # dense1 = layers.Dropout(0.25)(dense1)
    # dense2 = layers.Dense(256, activation="relu", name = "dense2")(dense1)
    dense2 = layers.BatchNormalization()(flatten)#(dense2)
    output = layers.Dense(256, name = "embedding_layer")(dense2)
    if norm:
        output = layers.Lambda(lambda x : tf.math.l2_normalize(x, axis = 1))(output)
    #output = layers.BatchNormalization()(output)
    return Model(backbone.input, output, name = "embedding")



if __name__ == "__main__":

    import wandb
    from wandb.keras import WandbCallback

    input_size = 380
    batch_size = 50
    max_epoch = 20
    ckpt_path = 'data/triplet_0/ckpt/V2S_triplet_0'
    secrets = make_secrets()



    train_df = pd.read_csv('data/triplet_0/train_df.csv', index_col=0)
    val_df = pd.read_csv('data/triplet_0/val_df.csv', index_col=0)
    df_kp = pd.read_csv('data/arcface_df_1/df_kp.csv', index_col=0)
    front_df = pd.read_csv('data/arcface_df_1/front_df.csv', index_col=0)

    front_df['property_id'] = -1
    train_df = pd.concat((train_df, front_df), ignore_index=True)

    train_ID = train_df.property_id.unique()
    val_ID = val_df.property_id.values

    train_input = DataGenerator(train_df, df_kp, batch_size, input_size, False, True)
    # val_images = load_violin_cropped(
    #     instance_id_list = val_df.instance_id.values,
    #     keypoints_df = df_kp,
    #     secrets = secrets,
    #     size = 380,
    #     square = False,
    #     augmentation = False,
    #     loading_bar=True,
    #     resize = None
    # )

    ## save/load validation images
    # np.save('data/triplet_0/val_imgs.npy', val_images)
    val_imgs = np.load('data/triplet_0/val_imgs.npy')

    print(f"train set: \nnumber of IDs: {train_ID.shape[0]} \nnumber of instances: {train_df.instance_id.unique().shape[0]}\n")
    print(f"validation set: \nnumber of IDs: {val_df.property_id.unique().shape[0]} \nnumber of instances: {val_df.instance_id.unique().shape[0]}\n")


    model = tf.keras.models.load_model('data/saved_models/arcface_V2S_embedding_.h5')
    for layer in model.layers[-5:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    #model = make_model(0, input_size, square=False, norm = False)
    #model.load_weights("data/model1/checkpoint5/checkpt_no_fine_tuning_resolution_aug.ckpt")

    model.compile(
        loss =SemiHardTripletLoss(), 
        optimizer=Adam(learning_rate = 1e-4),\
        # metrics = [
        #     DistanceAccuracy(type = "cosine"), 
        #     DistanceAccuracy(n_images=5, type = "cosine", name = "accuracy_5"), 
        #     DistanceAccuracy(n_images=10, type = "cosine", name = "accuracy_10")
        #  ]
         )


    
    wandb.init(project="triplet_loss", entity="padda")
    wandb.config = {
        "learning_rate": "1e-4 *0.5 each epoch",
        "epochs": max_epoch,
        "batch_size": batch_size,
        "loss": "semi-hard",
        "Offline hard mining" : "False",
        "augmentation" : "True"  
        }

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                 save_weights_only=True,
                                                 #save_best_only=True,
                                                 verbose=1)


    def scheduler(epoch, lr):
        if epoch >0 and epoch<=3:
            lr *= 0.5
        print("****** scheduler ****** ", lr)
        return lr
    schedule_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    #train_inputs = OfflineMiningInputGenerator(df_train, model, ckpt_path, max_epoch, 70, input_size, engine_, secrets_, False, True)  
    #train_inputs = DataGeneratorNPair(df_train, 70, input_size, engine = engine_, secrets=secrets_, square = False, augmentation=True, model=model)


    model.fit(\
        train_input, \
        epochs = max_epoch, \
        callbacks = [ \
            cp_callback, \
            ValidationMetricsCallback(
                val_IDs=val_ID,
                val_imgs=val_imgs,
                n_steps=40
            ), \
            WandbCallback(log_batch_frequency = 4), \
            schedule_callback
            ])

