import pandas as pd
import numpy as np
import cv2
import os
import sys
import yaml
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.utils import Sequence 
from violinmatch.utils.config import make_secrets
from violinmatch.utils.drive import retrieve_img
from violinmatch.utils.database import connect_db
from violinmatch.recognition.augmentation import contrast, crop_back_augmented
from violinmatch.recognition.metrics import pairwise_cosine_similarity




def display(image):

    cv2.namedWindow('AnnotationWindow', cv2.WINDOW_NORMAL)

    while True:

        cv2.imshow('AnnotationWindow', image)

        if (tt := cv2.waitKey(10) & int(0xFF)) != 255:

            if tt == ord('q'):

                cv2.destroyAllWindows()
                sys.exit()
            if tt == ord('n'):
                break

def resize_pad(image, size : int = 224, square : bool = True):
    """
    resizes the image to make it fit in the model.
    If square = True: resizes into a square of dimension size x size
    If square = False: resize into a rectangle of dimension size/4 x size*4: a rectangle with the same number of pixels as the square of size x size
    """
    h, w = image.shape[:2]

    if square:
        if h==w:
            return cv2.resize(image, (size, size))

        elif h > w:
            ratio = w/h
            short_size = round(size*ratio)
            
            image = cv2.resize(image, (short_size, size))
            return cv2.copyMakeBorder(image, top = 0, left = 0, bottom = 0, right = size-short_size, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

        else:

            ratio = h/w
            short_size = round(size*ratio)
            image = cv2.resize(image, (short_size, size))
            return cv2.copyMakeBorder(image, top = 0, left = 0, right = 0, bottom = size-short_size, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

    else:
        if h > (4*w):
            ratio = w/h
            new_w = round(2*size*ratio)
            
            image = cv2.resize(image, (new_w, int(2*size)))
            return cv2.copyMakeBorder(image, top = 0, left = 0, bottom = 0, right = int(size/2-new_w), borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
            
        elif h < (4*w):
            ratio = h/w
            new_h = round(size*ratio/2)
            image = cv2.resize(image, (int(size/2), new_h))
            image  = cv2.copyMakeBorder(image, top = 0, left = 0, right = 0, bottom = 2*size-new_h, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
            return(image)
            
        else:
            return cv2.resize(image, (int(size/2), int(size*2)))


# This function will only be useful if we'll use all back images for the negative part of the triplet. Otherwise it's better to use get_keypoints
def get_anchor_positive_negative_kps(engine = None, secrets = None, return_neg = False):
    """
    **OBSOLETE**

    if return_neg = True :  returns 2 pandas dataframe, the first one containing all of photo_keypoint table (negative) with only the c-out keypoints
                            and the second one containing only those with more than 1 photo per ID (positive and anchor)
    if return_neg = False:  returns only the dataframe for positive and anchor
    """

    if not secrets:
        secrets = make_secrets()
    if not engine:
        engine = connect_db(secrets)
    
    stt = """
    select p.ID, p.filename, p.is_private pk.photo_id, pk.keypoint_id, pk.x, pk.y
    from photo_keypoint pk
    left join photo p
    on pk.photo_id = p.img_id
    where pk.keypoint_id in (1, 2, 3, 4)
    """

    with engine.connect() as conn:

        all_pk = pd.read_sql(stt, conn)

        stt = f"""
        select p.ID, count(*) as img_count from photo
        where img_id in (select distinct(photo_id) from photo_keypoint)
        group by ID
        having img_count > 1
        """
        positive_IDs = pd.read_sql(stt, conn).ID.values

        if return_neg==True:
            return all_pk, all_pk.loc[all_pk.ID in positive_IDs, :].reset_index(drop=True).copy()

        return all_pk.loc[all_pk.ID in positive_IDs, :].reset_index(drop=True).copy()

def get_id_keypoints(engine = None, secrets = None):
    """
    **OBSOLETE**
    returns a pandas dataframe containing img_id, ID, filename, is_private, keypoint_id and keypoint *for IDs having more than 1 back image*
    """

    if not secrets:
        secrets = make_secrets()
    if not engine:
        engine = connect_db(secrets)
    
    with engine.connect() as conn:
        stt = f"""
        select ID, count(*) as img_count from photo
        where img_id in (select distinct(photo_id) from photo_keypoint)
        group by ID
        having img_count > 1
        """
        positive_IDs = pd.read_sql(stt, conn).ID.values

        stt = f"""
        select p.ID, p.filename, p.is_private, pk.photo_id, pk.keypoint_id, pk.x, pk.y
        from photo_keypoint pk
        left join photo p
        on pk.photo_id = p.img_id
        where p.ID in {tuple(positive_IDs)}
        and pk.keypoint_id in (1, 2, 3, 4)
        """

        return pd.read_sql(stt, conn)


def crop_back(df, secrets = None, square : bool = True):
    """
    takes an image of the back of a violin as an input, with the cbout keypoint locations, and returns a delimited area of the back of the violin.
    if square = True, this area is a square, if square = False, this area is a rectangle that takes most of the violin's back.
    all resizing/padding of the cropped image has to be done afterwards
    """
    if 'imguri' in df.columns:
        image = cv2.imread(df.imguri.values[0])
    else:
        if not secrets:
            secrets = make_secrets()
        df.reset_index(drop=True, inplace=True)
        ID = df.ID.unique()[0] #df.property_id.unique()[0]
        filename = df.filename.unique()[0]
        is_pvt = df.is_private.unique()[0]
        image = retrieve_img(ID = ID, filename = filename, is_pvt = is_pvt)

        
    kp1 = df.loc[df.keypoint_id == 1,['x', 'y']].values.squeeze()
    kp2 = df.loc[df.keypoint_id == 2,['x', 'y']].values.squeeze()
    kp3 = df.loc[df.keypoint_id == 3,['x', 'y']].values.squeeze()
    kp4 = df.loc[df.keypoint_id == 4,['x', 'y']].values.squeeze()
 

    (height, width) = image.shape[:2]
    
    # get c-bout angles:
    treb_cbout = kp1-kp2
    treb_cbout[1] *= -1
    treb_angle = np.angle(treb_cbout[0] + treb_cbout[1]*1j)

    bass_cbout = kp3-kp4
    bass_cbout[1] *= -1
    bass_angle = np.angle(bass_cbout[0] + bass_cbout[1]*1j)

    # np.angle renvoie un angle en radian allant de -pi à +pi. ça peut poser problème, mais pas avec ce qui suit
    if treb_angle*bass_angle<0:
        if abs(treb_angle)>np.pi/2 and abs(bass_angle)>np.pi/2:
            if treb_angle < 0:
                treb_angle = 2*np.pi + treb_angle
            if bass_angle < 0:
                bass_angle = 2*np.pi + bass_angle

    # rotation angle is the mean of the 2 angles

    angle = np.degrees((treb_angle+bass_angle)/2 )
    # angle_deg = np.degrees(angle)


    # get center (intersection of kp1-kp4 and kp2-kp3)

    center = (int(np.mean(np.array([kp1[0], kp2[0], kp3[0], kp4[0]]))), int(np.mean(np.array([kp1[1], kp2[1], kp3[1], kp4[1]]))))

    rotation_matrix = cv2.getRotationMatrix2D(center, 90-angle, 1)

    # cv2 ne se fatigue pas à resize les images correctement après une rotation. il faut faire cela manuellement. 
    # la nouvelle image sera un carré de la taille de la diagonale, au cas où le violon est en diagonale. 
    diag = int(np.sqrt(width**2+height**2))
    new_center = [diag//2, diag//2]
    # On veut que le centre du violon soit au centre de la nouvelle image. 
    # pour cela il faut modifier la 3e colonne de la matrice de rotation, qui se charge de la translation post-rotation
    (tx,ty) = ((new_center[0]-center[0]),(new_center[1]-center[1]))
    rotation_matrix[0,2] += tx 
    rotation_matrix[1,2] += ty
    rotated_image = cv2.warpAffine(src = image, M = rotation_matrix, dsize = (diag, diag))

    # apply rotation to keypoints
    locs = locals()
    for i in range(1,5):
        locs["kp"+str(i)] = np.dot(rotation_matrix[:, :2], locs["kp"+str(i)])+rotation_matrix[:,2]
        locs["kp"+str(i)] = locs["kp"+str(i)].astype(int)


    # calculate threequarter ratio:
    dist_treble = np.linalg.norm(treb_cbout) 
    dist_bass   = np.linalg.norm(bass_cbout)
    dist_mean = (dist_bass+dist_treble)/2
    threequarter_ratio = (dist_treble-dist_bass)/dist_mean
    # ratio negatif = bass devant, treble derrière. varie généralement entre 0 et 0.2. violon pris de face ~ 10e-3/-2 violon de 3/4 ~ >10e-1
    inter_bout = (1-2*threequarter_ratio**2)*np.linalg.norm(kp1 - kp3)

    #cropped image is a square or a rectangle?
    xmin = int(new_center[0] - 0.3*inter_bout)
    xmax = int(new_center[0] + 0.3*inter_bout)
    if square:
        x = xmax-xmin
        ymin = max(0,int(new_center[1] - x/2))
        ymax = ymin+x
    else:
        ymin = max(0, int(new_center[1] - dist_mean*1.5))
        ymax = int(new_center[1] + dist_mean*1.7)



    return rotated_image[ymin:ymax, xmin:xmax, :]

def load_violin_cropped(instance_id_list : np.ndarray, keypoints_df = None, secrets = None, size : int = 380, square : bool = True, augmentation : bool = False, loading_bar : bool = False, resize = 1e6):
    """takes an list of n img_ids and returns a array of cropped images of shape (n,size,size,3) (224 for efficientnetB0)
    image channels are in RGB
    """
    if not secrets:
        secrets = make_secrets()
        

    keypoints_df = keypoints_df.loc[keypoints_df.instance_id.isin(instance_id_list),:].copy()

    # initialize for loop
    if square:
        M_list = np.ones((1,size,size,3),np.uint8)
    else:
        M_list = np.ones((1,int(size*2),int(size/2),3),np.uint8)

    if loading_bar:
        instance_id_list = tqdm(instance_id_list)

    for iid in instance_id_list:
        df = keypoints_df.loc[keypoints_df.instance_id == iid,:]
        
        try:
            if augmentation:
                M = crop_back_augmented(df, secrets , square, resize)
            else:
                M = crop_back(df, secrets , square)
                # convert into black an white + contrast (CLAHE):
                M = cv2.cvtColor(M, cv2.COLOR_BGR2GRAY)
                M = np.stack((M,)*3, axis=-1)
                M = contrast(M, True)
            M = resize_pad(M, size, square)
            M = np.expand_dims(resize_pad(M, size, square),0)
            M_list = np.concatenate((M_list, M), axis = 0)
        except Exception as e:
            print(f"problem with instance no {iid}")
            raise e
            

    M_list = M_list[1:,:,:,:]
    return M_list


def get_pos_neg_index(anchor_ID : int, ID_list):
    """
    takes the anchor ID and the ID array containing all of the batch IDs. 
    returns 2 index arrays: positive and negative.
    """
    return np.argwhere(ID_list == anchor_ID), np.argwhere(ID_list != anchor_ID)

class DataGenerator(Sequence):
    def __init__(self, df : pd.DataFrame, keypoint_df : pd.DataFrame, batch_size : int = 50, image_size : int = 380, square : bool = False, augmentation : bool = True):
        """ df: subset (train, val or test) of the dataframe returned by get_id_keypoints()
        """
        
        self.square = square
        self.augmentation = augmentation
        self.img_size = image_size
        # shuffle df
        # batch the IDs, then get photo_ids from IDs for each batch
        self.df = df
        self.df_kp = keypoint_df
        self.IDs = self.df.property_id.unique() 
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
        tt = self.df.loc[self.df.property_id.isin(IDs), ["ID","photo_id"]]
        labels = tt.property_id.values
        img_ids = tt.instance_id.values
        img_list = load_violin_cropped(
            img_id_list= img_ids,
            keypoints_df= self.df_kp,
            size = 380, 
            square = False,
            augmentation= True,
            loading_bar= False,
            resize = 1e6
        )
        return img_list, labels

class DataGeneratorNPair(Sequence):
    def __init__(self, df : pd.DataFrame, model = None, batch_size : int = 50, image_size : int = 224 , engine = None, secrets = None, square : bool = True, augmentation : bool = False):
        """ df: subset (train, val or test) of the dataframe returned by get_id_keypoints()
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
        self.square = square
        self.augmentation = augmentation
        self.img_size = image_size
        # shuffle df
        # batch the IDs, then get photo_ids from IDs for each batch
        self.df = df
        self.IDs = self.df.ID.unique() 
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
        labels = []
        if self.square:
            img_list = np.ones((1,self.img_size,self.img_size,3),np.uint8)
        else:
            img_list = np.ones((1,int(self.img_size*2),int(self.img_size/2),3),np.uint8)

        for ID in IDs:
            labels.append(ID)
            labels.append(ID)
            img_ids = self.df.loc[self.df.ID == ID, 'photo_id'].unique()
            imgs = load_violin_cropped(img_ids, self.df, self.engine, self.secrets, self.img_size, self.square, self.augmentation)
            # if there are only 2 images, just add them to the img_list. 
            # if there are more than 2 images, take the 2 that are the further appart to one another in the embedding space
            if img_ids.shape[0]==2:
                img_list = np.concatenate((img_list, imgs), axis = 0)
            else:
                embeddings = self.model(imgs)
                distances = pairwise_cosine_similarity(embeddings)
                max_distance_arg = tf.argmax(tf.reshape(tf.cast(distances, tf.float64), -1))
                index =  np.unravel_index(max_distance_arg, shape = (img_ids.shape[0],)*2)
                img_list = np.concatenate((img_list, np.expand_dims(imgs[index[0]], 0)), axis = 0)
                img_list = np.concatenate((img_list, np.expand_dims(imgs[index[1]], 0)), axis = 0)


        return img_list[1:], np.array(labels)



if __name__=="__main__":


    sys.exit()
    