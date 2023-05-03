from violinmatch.utils.config import make_secrets
from violinmatch.utils.drive import retrieve_img
from violinmatch.utils.database import connect_db
import tensorflow as tf
from math import sqrt
import cv2
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import random



def display(image, img_id=None):

    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/bad_prediction.csv"):
        df = pd.DataFrame(columns=["photo_id"])
        df.to_csv("data/bad_prediction.csv")

    cv2.namedWindow('AnnotationWindow', cv2.WINDOW_NORMAL)

    while True:

        cv2.imshow('AnnotationWindow', image)

        if (tt := cv2.waitKey(10) & int(0xFF)) != 255:

            
            if tt == ord('n'):
                break
            # if tt == ord('b'):
            #     df = pd.read_csv("data/bad_prediction.csv", index_col=0)
            #     temp = pd.DataFrame([img_id], columns = ["photo_id"])
            #     df = pd.concat([df, temp], ignore_index=True)
            #     df.to_csv("data/bad_prediction.csv")
            #     break
            if tt == ord('q'):

                cv2.destroyAllWindows()
                sys.exit()


def brightness(img, low = 0.7, high = 1.3):
    value = random.uniform(low, high*2)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    # hsv[:,:,1] = hsv[:,:,1]*value
    # hsv[:,:,1][hsv[:,:,1]>255]  = 255
    if random.randint(0,1):
        hsv[:,:,2] = hsv[:,:,2]//(value*2) 
    else:
        hsv[:,:,2] = hsv[:,:,2]*value 
        hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def noisy(image):
    mean = 0
    var = random.randint(30,70)
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,image.shape)
    gauss = gauss.reshape(*image.shape)
    noisy = image + gauss
    noisy = noisy.astype(np.uint8)
    noisy[noisy>255] = 255
    noisy[noisy<0] = 0
    return noisy



def contrast(img, clahe : bool = True):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel-------------------------------------------
    if clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
    else:
        cl = cv2.equalizeHist(l)
    
    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.stack((img,)*3, axis=-1) #stack 3 times the grayscale image in order to still have 3 channels

def reduce_resolution(img, factor = None):
    h, w = img.shape[:2]
    if not factor:
        factor = float(random.randint(3,5))/10.
    img = cv2.resize(img, (int(w*factor), int(h*factor)))   #  (img, fx=factor, fy=factor)
    return img#cv2.resize(img, (w,h))

def downsize(img, size = 1e6):
    """
    returns an resized image with the same aspect ratio with width * height <= size
    """
    (h, w) = img.shape[:2]
    if h*w>size:
        new_w = int(sqrt(size * w/h))
        new_h = round(new_w*h/w)
        img = cv2.resize(img,(new_w, new_h))
        return img
    else:
        return img

def flash_reflexion(img):
    
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    
    angle = random.randint(0,360)
    center = (random.randint(0, w), random.randint(0, h))
    axe1 = random.randint(w//20, w//4)
    axe2 = random.randint(h//20, h//4)
    increment = h*w//100000+1
    while axe1>0 and axe2>0:

        reflexion = np.zeros_like(img,np.uint8)
        reflexion = cv2.ellipse(reflexion, center,\
            (axe1, axe2), angle, 0, 360, (2,2,2), -1).astype(np.float64)
        reflexion[reflexion==2] = 20
        reflexion = reflexion[:,:,0]
    
        hsv[:,:,1] = hsv[:,:,1] - reflexion
        hsv[:,:,2] = hsv[:,:,2] + reflexion
        axe1 -= increment
        axe2 -= increment
        if np.any(hsv>550):
                axe1 = axe2 = 0
    hsv[hsv>255] = 255
    hsv[hsv<0] = 0
    hsv = hsv.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return img

def side_reflexion(img):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    angle = 0
    axe2 = random.randint(h//20, h//3) 
    axe1 = random.randint(w//20, max(w//20+1,min(w//2,axe2//3)))
    thickness_ = random.randint(w//8, w//4)

    center_w = [[0,],[w,],[0,w]]
    center_w = center_w[random.randint(0,2)]
    for cw in center_w:
        center = (cw, h//2)

        thickness = thickness_

        increment = w//30
        while thickness>0:
            reflexion = np.zeros_like(img,np.uint8)
            reflexion = cv2.ellipse(
                reflexion, 
                center,
                (axe1,axe2),#,(w//3, h//20), 
                angle, 
                0, 
                360, 
                (2,2,2), 
                thickness
                ).astype(np.float64)
            reflexion[reflexion==2] = 20
            reflexion = reflexion[:,:,0]
            thickness-=increment
            hsv[:,:,1] = hsv[:,:,1] - reflexion
            hsv[:,:,2] = hsv[:,:,2] + reflexion
            if np.any(hsv>650):
                thickness = 0

    hsv[hsv>255] = 255
    hsv[hsv<0] = 0
    hsv = hsv.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img

def dots(img):

    h, w = img.shape[:2]
    dot_count =  random.randint(3, 15)

    # whit or black
    if random.randint(0,1):
        c = random.randint(0,50)  
    else:
        c = random.randint(205, 255)

    color = (c, c, c)
    max_size = max(2, h*w//50000)
    m = max_size*20
    min_size = 1 # max(1, max_size-3)
    for _ in range(dot_count):

        
        x = random.randint(0, w)
        y = random.randint(0, h)
        size = random.randint(min_size, max_size)
        if random.randint(0,1):
            img = cv2.circle(img, (x, y), size, color, -1)
        else:
            
            hh = random.randint(-m,m)
            ww = random.randint(-m,m)
            x2 = max(0,x+ww)
            y2 = max(0,y+hh)
            img = cv2.line(img, (x,y), (x2, y2), color, random.randint(1,2))
            
    return img

def hide_quarter(img, quarter : int = None):
    h, w = img.shape[:2]
    if not quarter:
        quarter = random.randint(1,4)
    if quarter == 1:
        img[0:h//2, 0:w//2]=0
    elif quarter == 2:
        img[0:h//2, w//2:]=0
    elif quarter == 3:
        img[h//2:, 0:w//2]=0
    elif quarter == 4:
        img[h//2:, w//2:]=0
    return img

def blur(img):
    h,w = img.shape[:2]
    img_size =  h*w//1000
    if img_size<100:
        ksize = 1
    elif img_size<200:
        ksize = 5
    elif img_size<300:
        ksize = 8
    elif img_size<400:
        ksize = 10
    else:
        ksize = 13
    kernel_size = random.randint(1,ksize)*2+1
    return cv2.GaussianBlur(img, (kernel_size,kernel_size), 0)

def augmentation_pipeline(image):

    
    image = contrast(image, clahe=True)
    image = dots(image)
    r = random.randint(0,4)
    if r in (0,1):
        image = flash_reflexion(image)
    elif r==2:
        image = side_reflexion(image)
    
    image = brightness(image)

    r = random.randint(0,5)
    if r==0:
        image = tf.image.random_jpeg_quality(image, 2,20)
        image = image.numpy().astype(np.uint8)
    elif r==1:
        image = noisy(image)
    elif r in (2,3):
        image = blur(image)

    image = grayscale(image)

    
    if not random.randint(0,4): #horizontal flip
        image = image[:,::-1,:]
    if not random.randint(0,4): #inverse colors
        image = image[:,:,::-1]
    return image

def crop_back_augmented(df, secrets = None, square : bool = True, resize = None):
    """
    takes an image of the back of a violin as an input, with the cbout keypoint locations, and returns a delimited area of the back of the violin.
    if square = True, this area is a square, if square = False, this area is a rectangle thate takes most of the violin's back.
    all resizing/padding of the cropped image has to be done afterwards
    """
    df = df.reset_index(drop=True)
    if 'imguri' in df.columns:
        image = cv2.imread(df.imguri.values[0])
    else:
        if not secrets:
            secrets = make_secrets()
        ID = df.ID.unique()[0] #df.property_id.unique()[0]
        filename = df.filename.unique()[0]
        is_pvt = df.is_private.unique()[0]
        image = retrieve_img(ID = ID, filename = filename, is_pvt = is_pvt)

    
    kp1 = df.loc[df.keypoint_id == 1,['x', 'y']].values.squeeze()
    kp2 = df.loc[df.keypoint_id == 2,['x', 'y']].values.squeeze()
    kp3 = df.loc[df.keypoint_id == 3,['x', 'y']].values.squeeze()
    kp4 = df.loc[df.keypoint_id == 4,['x', 'y']].values.squeeze()

    
    height, width = image.shape[:2]
    height_0, width_0 = (height, width)
    if resize:
        image = downsize(image, resize)
        height, width = image.shape[:2]
        ratio = np.array([width/width_0, height/height_0])
        kp1 = np.rint((kp1*ratio).astype(np.float64))
        kp2 = np.rint((kp2*ratio).astype(np.float64))
        kp3 = np.rint((kp3*ratio).astype(np.float64))
        kp4 = np.rint((kp4*ratio).astype(np.float64))
    
    # get c-bout angles:
    treb_cbout = kp1-kp2
    treb_cbout[1] *= -1
    treb_angle = np.angle(treb_cbout[0] + treb_cbout[1]*1j)

    bass_cbout = kp3-kp4
    bass_cbout[1] *= -1
    bass_angle = np.angle(bass_cbout[0] + bass_cbout[1]*1j)

    # angle goes from -pi to +pi. it can cause problems around pi, if angles are of opposite signs
    if treb_angle*bass_angle<0:
        if abs(treb_angle)>np.pi/2 and abs(bass_angle)>np.pi/2:
            if treb_angle < 0:
                treb_angle = 2*np.pi + treb_angle
            if bass_angle < 0:
                bass_angle = 2*np.pi + bass_angle

    # rotation angle is the mean of the 2 angles
    
    angle = np.degrees((treb_angle+bass_angle)/2)
    # angle augmentation
    #angle = angle + random.uniform(-1, 1)
    

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
        ymin = int(new_center[1] - x/2)
        ymax = ymin+x
    else:
        ymin = int(new_center[1] - dist_mean*1.5)
        ymax = int(new_center[1] + dist_mean*1.7)
    
    # augmentation of the crop
    y_eps = round((ymax-ymin)*0.01)
    y_eps = random.randint(-y_eps, y_eps)
    ymin = max(0,ymin-y_eps)
    ymax = min(ymax+y_eps, rotated_image.shape[0])

    x_eps = round((xmax-xmin)*0.05)
    x_eps = random.randint(-x_eps, x_eps)
    xmin = max(0,xmin-x_eps)
    xmax = min(xmax+x_eps, rotated_image.shape[1])

    # either random crop or center around the middle of the violin, overwrite ymin and ymax ####
    if not random.randint(0,4):
        ymin = int((locs['kp1'][1] + locs['kp3'][1])/2 - inter_bout/6)
        ymin = max(0,ymin)
        ymax = int((locs['kp2'][1] + locs['kp4'][1])/2 + inter_bout/6)
        ymax = min(ymax, rotated_image.shape[0])
    #     else: # random crop
    #         mmin = int((locs['kp1'][1] + locs['kp3'][1])/2)
    #         mmax = int((locs['kp2'][1] + locs['kp4'][1])/2)
    #         y_diff = mmax - mmin
    #         ymin = random.randint(ymin, ymax-y_diff)
    #         ymin = max(0,ymin)
    #         ymax = ymin+y_diff
    #         ymax = min(ymax, rotated_image.shape[0])
        


    image = rotated_image[ ymin:ymax , xmin:xmax, :]
    return augmentation_pipeline(image)



if __name__ == "__main__":

    def resize_pad(image, size : int = 380, square : bool = False):
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

    secrets = make_secrets()
    train_df = pd.read_csv('data/arcface_df_1/train_df.csv', index_col=0)
    df_kp = pd.read_csv('data/arcface_df_1/df_kp.csv', index_col=0)
    iid_list = np.load("data/horizontal_img.npy")#train_df.instance_id.unique()
    np.random.shuffle(iid_list)
    print(iid_list.shape)
    for iid in iid_list:
        image_path = train_df.loc[train_df.instance_id == iid, 'imguri'].values[0]
        image = cv2.imread(image_path)
        for i in range(4):
            x = df_kp.loc[(df_kp.keypoint_id==i+1) & (df_kp.instance_id == iid), "x"].values[0]
            y = df_kp.loc[(df_kp.keypoint_id==i+1) & (df_kp.instance_id == iid), "y"].values[0]
            cv2.circle(image, (x,y), 30, (0,15+i*50,0), -1)


        display(image)
        
        _ = df_kp.loc[df_kp.instance_id == iid]
        for i in range(2):
            display(resize_pad(crop_back_augmented(_, secrets, False, None)))


