
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import backend as keras
from glob import glob

# =============================================================================
#  Set directories for ROI segmentation
# =============================================================================

model_path = 'model'
source_dir = 'dataset'
segmentation_dir = os.path.join(source_dir,'segmentation')
segmentation_result_dir = os.path.join(source_dir,'segmentation_result')

# =============================================================================
#  Custom Objects 
# =============================================================================

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

# =============================================================================
#  Custom loss function
# =============================================================================

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# =============================================================================
#  resizing, normalization and reshape of images
# =============================================================================

def load_image(image_files, target_size = (256,256)):
    img = cv2.imread(image_files, cv2.IMREAD_GRAYSCALE)
    img = img / 255
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img,(1,) + img.shape)
    return img

# =============================================================================
#  image generator pipe line
# =============================================================================

def image_generator(image_files, target_size = (256,256)):
    for image_files in image_files:
        yield load_image(image_files, target_size)
        
# =============================================================================
#  resize, find contours, extract ROI using bounding box and saved files 
#  saved image size = 224 x 224
# =============================================================================
              
def save_result(save_path, npyfile, image_files):
    
    for i, item in enumerate(npyfile):
        result_file = image_files[i]
        X_shape = 512
        im = cv2.resize(cv2.imread(result_file),(X_shape,X_shape))[:,:,0]       
        img = (item[:, :, 0] * 255.).astype(np.uint8)
        ret, thresh = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY)
        contours, hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        boxes = []        
        for c in cnt:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x,y, x+w,y+h])
        boxes = np.asarray(boxes)
        left, top = np.min(boxes, axis=0)[:2]
        right, bottom = np.max(boxes, axis=0)[2:]
        cropped = im[top:(bottom +50), left:(right+50)]
        cropped=cv2.resize(cropped,(224,224),interpolation=cv2.INTER_CUBIC)
        boxes=boxes.tolist()
        boxes.clear()
        
        filename, fileext = os.path.splitext(os.path.basename(result_file))

        result_file = os.path.join(save_path,"%s%s" % (filename, fileext))

        cv2.imwrite(result_file, cropped)


# =============================================================================
#  Load saved model from directory
# =============================================================================

model_path = model_path
model = tf.keras.models.load_model(model_path,
        custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef}) # load model  

# =============================================================================
#  prepare input files for prediction
# =============================================================================
      
image_files = glob(os.path.join(segmentation_dir, "*.*"))
image_generator = image_generator(image_files, target_size=(512,512))

# =============================================================================
#  ROI prediction
# =============================================================================

results = model.predict(image_generator, len(image_files), verbose=1)

# =============================================================================
#  save ROI
# =============================================================================

save_result(segmentation_result_dir, results, image_files)
