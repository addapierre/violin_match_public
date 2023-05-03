import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

import numpy as np
import yaml
import sys

from violinmatch.utils.database import connect_db
from violinmatch.recognition.input_generator import get_id_keypoints, load_violin_cropped
from violinmatch.recognition.metrics import pairwise_distances, pairwise_cosine_similarity
from violinmatch.recognition.triplet_utils import get_anchor_positive_triplet_mask, get_anchor_negative_triplet_mask, get_triplet_mask
from tqdm import tqdm





def batch_hard(labels, embeddings, margin = 0.2, squared : bool = False):

    pairwise_dist = pairwise_distances(embeddings, squared=squared)
    max_dist = tf.reduce_max(pairwise_dist) # will be used for the negative mask: we're only interested in the lowest negative values
    # each anchor has a single positive and a single negative attributed to it
    # positive vector
    pos_mask = get_anchor_positive_triplet_mask(labels)
    pos_mask = tf.cast(pos_mask,tf.float32)
    pos_vector = tf.multiply(pos_mask, pairwise_dist)
    pos_vector = tf.reduce_max(pos_vector, 1)
    # negative vector
    
    neg_mask = get_anchor_negative_triplet_mask(labels)
    neg_mask = tf.cast( tf.logical_not(neg_mask), tf.float32)
    neg_mask *= max_dist
    neg_vector = pairwise_dist+neg_mask
    neg_vector = tf.reduce_min(neg_vector, 1)

    # triplet loss
    triplet_loss = pos_vector - neg_vector + margin
    triplet_loss = tf.maximum(triplet_loss, 0.)
    triplet_loss = tf.reduce_mean(triplet_loss) 
    return triplet_loss


def batch_semi_hard(labels, embeddings, margin = 0.2, squared : bool = False):
    """
    Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones (semi-hard mining).
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    pairwise_dist = pairwise_distances(embeddings, squared=squared)
    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = get_triplet_mask(labels)
    mask = tf.cast(mask, tf.float32)
    triplet_loss = tf.multiply(mask, triplet_loss)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    #num_valid_triplets = tf.reduce_sum(mask)
    #fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.cast(tf.cast(tf.reduce_sum(triplet_loss), tf.float64) / tf.cast(tf.maximum(num_positive_triplets , 1e-16), tf.float64), tf.float32)
    
    return triplet_loss

def n_pair_loss(labels, embeddings, reg_lambda=0.002):
    embeddings_anchor = embeddings[::2]
    embeddings_positive = embeddings[1::2]
    reg_anchor = tf.reduce_mean(
      tf.reduce_sum(tf.square(embeddings_anchor), 1))
    reg_positive = tf.reduce_mean(
      tf.reduce_sum(tf.square(embeddings_positive), 1))
    l2loss = tf.multiply(
      0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')

    similarity_matrix = tf.matmul(
      embeddings_anchor, embeddings_positive, transpose_a=False,
      transpose_b=True)

    lshape = tf.shape(embeddings_anchor)[0]
    labels = tf.eye(lshape, lshape)
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = similarity_matrix)
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)

    return l2loss + cross_entropy_loss

class NPairLoss(tf.keras.losses.Loss):

    def __init__(self, reg_lambda=0.002, name = "NPair_loss"):
        super().__init__(name=name)
        self.reg_lambda = reg_lambda


    def call(self, y_true, y_pred):
        return n_pair_loss(y_true, y_pred, reg_lambda = self.reg_lambda)

class SemiHardTripletLoss(tf.keras.losses.Loss):

    def __init__(self, margin = 0.2, squared : bool = False, name = "triplet_loss"):
        super().__init__(name=name)
        self.margin = margin
        self.squared = squared

    def call(self, y_true, y_pred):
        return batch_semi_hard(y_true, y_pred, margin = self.margin)

class HardTripletLoss(tf.keras.losses.Loss):

    def __init__(self, margin = 0.2, squared : bool = False, name = "triplet_loss"):
        super().__init__(name=name)
        self.margin = margin
        self.squared = squared

    def call(self, y_true, y_pred):
        return batch_hard(y_true, y_pred, margin = self.margin, squared=self.squared)


if __name__ == "__main__":

    sys.exit()