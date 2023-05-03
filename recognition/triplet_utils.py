import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import silhouette_score




def get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask

def get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True if a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_not_equal = tf.logical_not(tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool))

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True if a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask

def get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    
    indices_not_equal = tf.logical_not(tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool))
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


class ValidationMetricsCallback(Callback):
    def __init__(self, val_images, val_IDs, n_step=0):
        """
        Validation metric callback for triplet loss.
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

        embeddings = tf.expand_dims(tf.zeros(self.model.layers[-1].output_shape[1:]), 0)
        for batch in tqdm(self.batch_idx):
            M = self.model(self.val_imgs[batch])
            embeddings = tf.concat([embeddings, M], 0)
        return embeddings[1:]
        
    def get_accuracy(self, labels, embeddings):

        labels = tf.squeeze(labels)
        pairwise_dist = pairwise_cosine_similarity(embeddings)
        max_dist = tf.reduce_max(pairwise_dist)
        mask = tf.cast(tf.eye(tf.shape(labels)[0]), tf.float32)
        mask *= max_dist
        result = pairwise_dist+mask

        result = tf.argsort(tf.cast(result, tf.float64), axis = 1, direction="ASCENDING")[:,:10]
        result = tf.map_fn(lambda x: tf.gather(params = tf.squeeze(labels), indices=x), result, fn_output_signature = tf.int64)
        result = tf.transpose(result, (1,0))
        accuracy = tf.equal(result, labels)
        accuracy = tf.cast(tf.transpose(accuracy, (1,0)), tf.float32)
        accuracy1 = tf.reduce_mean(tf.reduce_max(accuracy[:,:1], axis = 1))
        accuracy5 = tf.reduce_mean(tf.reduce_max(accuracy[:,:5], axis = 1))
        accuracy10 = tf.reduce_mean(tf.reduce_max(accuracy, axis = 1) )    
        return accuracy1, accuracy5, accuracy10


    def on_train_batch_end(self, batch, logs = None):

        if self.step%10==0:
            embeddings = self.make_embeddings()
            accuracy1, accuracy5, accuracy10 = self.get_accuracy(self.labels, embeddings)
            logs["val_acc1"] = accuracy1
            logs["val_acc5"] = accuracy5
            logs["val_acc10"] = accuracy10
            logs["silhouette_score"] = silhouette_score(embeddings, self.labels)
        self.step +=1


    def on_epoch_end(self, epoch, logs = None):
        
        embeddings = self.make_embeddings()
        accuracy1, accuracy5, accuracy10 = self.get_accuracy(self.labels, embeddings)
        logs["val_acc1"] = accuracy1
        logs["val_acc5"] = accuracy5
        logs["val_acc10"] = accuracy10
        logs["silhouette_score"] = silhouette_score(embeddings, self.labels)
        self.step = 1
        