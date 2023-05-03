import tensorflow as tf
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from violinmatch.recognition.triplet_utils import get_triplet_mask


def pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0

        mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

def pairwise_cosine_similarity(embeddings, norm : bool = True):
    """Compute the 2D matrix of cosine similarity between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
    Returns:
        tensor of shape (batch_size, batch_size)
    """
    if norm:
        embeddings = tf.math.l2_normalize(embeddings, axis = 1)
    return 1 - tf.matmul(embeddings, embeddings, transpose_b=True)


def accuracy(labels, embeddings, n_images = 1, type = "euclidian"):
    """
    metrique: vérifie pour chaque image du batch si il y a au moins une image avec le même ID parmis les n embeddings d'images les plus proches en distance euclidienne.
    """
    labels = tf.squeeze(labels)
    if type == "euclidian":
        pairwise_dist = pairwise_distances(embeddings, squared=False)
    elif type == "cosine":
        pairwise_dist = pairwise_cosine_similarity(embeddings)
    else:
        raise(f'type must be either "euclidian" or "cosine, you entered {type}')
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
     
    return tf.reduce_mean(accuracy)


def fraction_positive_triplets(labels, embeddings, margin = 0.2, squared : bool = False):
    """
    Metric: Computes the fraction of valid triplet (i.e. easy triplets) over the whole batch.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        Valid triplet fraction as tf.float32
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
    num_valid_triplets = tf.reduce_sum(mask)
    return tf.cast(tf.cast(num_positive_triplets, tf.float64) / tf.cast(tf.maximum(num_valid_triplets, 1e-16), tf.float64), tf.float32)
    
        
class FractionPositiveTriplets(tf.keras.metrics.Metric):
    def __init__(self, margin = 0.2, name = "positive_triplet_ratio", **kwargs):
        super(FractionPositiveTriplets, self).__init__(name=name, **kwargs)
        self.margin = margin
        self.losses_ratio = self.add_weight(name='positive_triplet_ratio', initializer='zeros')


    def update_state(self, y_true, y_pred, sample_weight=None):
        self.reset_state()
        self.losses_ratio.assign_add(fraction_positive_triplets(y_true, y_pred, self.margin))

    def result(self):
        return self.losses_ratio

class DistanceAccuracy(tf.keras.metrics.Metric):
    def __init__(self, n_images = 1, type : str = "euclidian",  name = "accuracy_1", **kwargs):
        super(DistanceAccuracy, self).__init__(name=name, **kwargs)
        self.n_images = n_images
        self.type = type
        self.accuracy = self.add_weight(name='acc_1', initializer='zeros')


    def update_state(self, y_true, y_pred, sample_weight=None):
        self.reset_state()
        self.accuracy.assign_add(accuracy(y_true, y_pred, n_images=self.n_images,type=self.type))

    def result(self):
        return self.accuracy


    
    

  