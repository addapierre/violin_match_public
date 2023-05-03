import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from tensorflow import keras
from math import pi
from tqdm import tqdm
from sklearn.metrics import silhouette_score

from violinmatch.recognition.input_generator import display, crop_back, resize_pad
from violinmatch.recognition.augmentation import crop_back_augmented, contrast
from violinmatch.recognition.metrics import pairwise_cosine_similarity
from violinmatch.utils.config import make_secrets
from violinmatch.recognition.input_generator import load_violin_cropped


from tensorflow.keras.utils import Sequence 

class L2Normalization(keras.layers.Layer):
    """This layer normalizes the inputs with l2 normalization."""

    def __init__(self, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs):
        inputs = tf.nn.l2_normalize(inputs, axis=1)

        return inputs

    def get_config(self):
        config = super().get_config()
        return config

class ArcLayer(keras.layers.Layer):
    """Custom layer for ArcFace.
    This layer is equivalent a dense layer except the weights are normalized.
    """

    def __init__(self, units, kernel_regularizer=None, **kwargs):
        super(ArcLayer, self).__init__(**kwargs)
        self.units = units
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=[input_shape[-1], self.units],
                                      dtype=tf.float32,
                                      initializer=keras.initializers.HeNormal(),
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      name='kernel')
        self.built = True

    @tf.function
    def call(self, inputs):
        weights = tf.math.l2_normalize(self.kernel, axis=0)
        return tf.matmul(inputs, weights)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units,
                       "kernel_regularizer": self.kernel_regularizer})
        return config

class ArcLoss(keras.losses.Loss):
    """Additive angular margin loss.
    Original implementation: https://github.com/luckycallor/InsightFace-tensorflow
    """

    def __init__(self, margin=0.5, scale=64, name="arcloss"):
        """Build an additive angular margin loss object for Keras model."""
        super().__init__(name=name)
        self.margin = margin
        print(f'** margin : {self.margin} **')
        self.scale = scale
        self.threshold = tf.math.cos(pi - margin)
        self.cos_m = tf.math.cos(margin)
        self.sin_m = tf.math.sin(margin)

        # Safe margin: https://github.com/deepinsight/insightface/issues/108
        self.safe_margin = self.sin_m * margin

    @tf.function
    def call(self, y_true, y_pred):

        # Calculate the cosine value of theta + margin.
        #tf.assert_greater(y_pred,0.0, "\n**** y_pred inférieur à 0 ****\n")
        cos_t = y_pred
        sin_t = tf.math.sqrt(1 - tf.math.square(cos_t))

        cos_t_margin = tf.where(cos_t > self.threshold,
                                cos_t * self.cos_m - sin_t * self.sin_m,
                                cos_t - self.safe_margin)

        # The labels here had already been onehot encoded.
        mask = y_true
        cos_t_onehot = cos_t * mask
        cos_t_margin_onehot = cos_t_margin * mask

        # Calculate the final scaled logits.
        logits = (cos_t + cos_t_margin_onehot - cos_t_onehot) * self.scale

        losses = tf.nn.softmax_cross_entropy_with_logits(y_true, logits)

        return losses

    def get_config(self):
        config = super(ArcLoss, self).get_config()
        config.update({"margin": self.margin, "scale": self.scale})
        return config

class ValidationMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_images, val_IDs, n_step=0):
        """
        Validation metric callback for arcface.
        Arg: 
          val_images: Array containing the already preprocessed validation images.
          val_IDs: array containing the labels corresponding to the validation images
          n_step: integer. validation is performed every n_step batches. 
            used if the user want to monitor the validation during an epoch. 
            default: 0 => validation only at the end of epochs.
        """
        super(ValidationMetricsCallback, self).__init__()
        self.val_img = val_images
        self.labels = val_IDs
        self.n_step = n_step

        # embeddings are computed batch-wise
        batch_size = 50
        num_batches = np.ceil(self.labels.shape[0] / batch_size)
        self.batch_idx = np.array_split(range(self.labels.shape[0]), num_batches)

    def make_embeddings(self):

        # In arcface, the embeddings are the output from the penultimate layer. We named this layer 'embedding' when building the model.
        model = tf.keras.Model(
            inputs = self.model.input, 
            outputs = self.model.get_layer('embedding').output
            )
        # initiate a tensor with the right shape. Embeddings will be concatenated to that tensor
        embeddings = tf.expand_dims(tf.zeros(model.layers[-1].output_shape[1:]), 0)
        # remove tqdm to get rid of loading bar
        for batch in tqdm(self.batch_idx):
            imgs = self.val_img[batch]
            M = model(imgs)
            embeddings = tf.concat([embeddings, M], 0)
        return embeddings[1:]

    def get_accuracy(self, labels, embeddings):
      
      """
      Computes rank-N accuracy for deep metric learning.
      Args:
          labels: array of shape (batch_size,). Labels associated to the embeddings
          embeddings: tensor of shape (batch_size, embed_dim)
      Returns:
          (float, float, float) rank 1, 5, and 10 accuracies
      """
      labels = tf.squeeze(labels)
      pairwise_dist = pairwise_cosine_similarity(embeddings)

      # We need to add a mask to the pairwise_dist tensor, so the lowest distance is not the embedding with itself.
      max_dist = tf.reduce_max(pairwise_dist)
      mask = tf.cast(tf.eye(tf.shape(labels)[0]), tf.float32)
      mask *= max_dist
      # the cast in tf.float64 is to prevent strange behavior during the argsort, likely caused by arm64 architecture in graph mode.
      pairwise_dist_masked = tf.cast(pairwise_dist+mask, tf.float64)

      #argsort: select the 10 lowest distances' position for each line
      result = tf.argsort(pairwise_dist_masked, axis = 1, direction="ASCENDING")[:,:10]

      # seek labels corresponding to those distances with tf.gather
      result = tf.map_fn(lambda x: tf.gather(params = tf.squeeze(labels), indices=x), result, fn_output_signature = tf.int64)
      result = tf.transpose(result, (1,0))
      # compare them to the label array, returns 1 or 0
      accuracy = tf.equal(result, labels)
      # cast back to tf.float32
      accuracy = tf.cast(tf.transpose(accuracy, (1,0)), tf.float32)
      accuracy1 = tf.reduce_mean(tf.reduce_max(accuracy[:,:1], axis = 1))
      accuracy5 = tf.reduce_mean(tf.reduce_max(accuracy[:,:5], axis = 1))
      accuracy10 = tf.reduce_mean(tf.reduce_max(accuracy, axis = 1) )    
      return accuracy1, accuracy5, accuracy10


    def on_train_batch_end(self, batch, logs = None):
        if self.n_step != 0:
            if batch%self.n_step==0:

                embeddings = self.make_embeddings()
                
                accuracy1, accuracy5, accuracy10 = self.get_accuracy(self.labels, embeddings)
                logs["val_acc1"] = accuracy1
                logs["val_acc5"] = accuracy5
                logs["val_acc10"] = accuracy10

                #silhouette score is calculated using the sklearn function
                logs["silhouette_score"] = silhouette_score(embeddings, self.labels)

    def on_epoch_end(self, epoch, logs = None):
        
        embeddings = self.make_embeddings()
        accuracy1, accuracy5, accuracy10 = self.get_accuracy(self.labels, embeddings)
        logs["val_acc1"] = accuracy1
        logs["val_acc5"] = accuracy5
        logs["val_acc10"] = accuracy10
        logs["silhouette_score"] = silhouette_score(embeddings, self.labels)

    def on_epoch_begin(self, epoch, logs = None):
        
        if epoch == 0:
            print("epoch :",epoch)
            embeddings = self.make_embeddings()

            accuracy1, accuracy5, accuracy10 = self.get_accuracy(self.labels, embeddings)
            logs["val_acc1"] = accuracy1
            logs["val_acc5"] = accuracy5
            logs["val_acc10"] = accuracy10
            logs["silhouette_score"] = silhouette_score(embeddings, self.labels)   

class TrainDataGenerator(Sequence):
    def __init__(self, df : pd.DataFrame, keypoint_df : pd.DataFrame, one_hot_encoder, batch_size : int = 50, image_size : int = 380, secrets = None, square : bool = False, augmentation : bool = True):
        """ df: subset (train, val or test) of the dataframe returned by get_id_keypoints()
        """
        if not secrets:
            secrets = make_secrets()
        self.secrets = secrets
        self.square = square
        self.augmentation = augmentation
        self.img_size = image_size
        self.df = df
        self.df_kp = keypoint_df
        self.IDs = self.df.ID.unique() 
        self.ohe = one_hot_encoder
        np.random.shuffle(self.IDs)

        self.batch_size = batch_size
        self.num_batches = np.ceil(self.IDs.shape[0] / batch_size)
        self.batch_idx = np.array_split(range(self.IDs.shape[0]), self.num_batches)

    def on_epoch_end(self):
        np.random.shuffle(self.IDs)

    def __len__(self):
        return len(self.batch_idx)

    def __getitem__(self, idx):

        IDs = self.IDs[self.batch_idx[idx]]
        tt = self.df.loc[self.df.ID.isin(IDs), ["ID","photo_id"]]
        img_ids = tt.photo_id.values
        labels = self.ohe.transform(np.expand_dims(tt.ID.values,1)).toarray()
        img_list = load_violin_cropped(
            img_ids,
            self.df_kp,
            secrets = self.secrets,
            size = self.img_size,
            square=self.square,
            augmentation=self.augmentation,
            resize = 1e6
        )
        return img_list, labels




if __name__ == "__main__":
    secrets = make_secrets()
    train_df = pd.read_csv('data/train_df.csv', index_col=0)
    df_kp = pd.read_csv('data/df_kp.csv', index_col=0)

    photo_id_batch = train_df.photo_id.unique()[:40]
    df_kp = df_kp.loc[df_kp.photo_id.isin(photo_id_batch)]
    img_list = load_violin_cropped(
        photo_id_batch, 
        df_kp, 
        secrets = secrets, 
        size = 380, 
        square = False, 
        augmentation=True, 
        loading_bar=True)

    for i in range(img_list.shape[0]):
        img = img_list[i]
        display(img)
