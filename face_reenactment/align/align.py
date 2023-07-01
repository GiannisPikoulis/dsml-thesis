import imutils
import dlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import OrderedDict
import os, sys
from tqdm import tqdm
import cv2
import csv
import numpy as np
import argparse
import PIL
from skimage.io import imread, imsave
from skimage.transform import rescale, estimate_transform, warp, resize
import scipy
from abc import abstractmethod, ABC
import torch
import pickle as pkl
import face_alignment


FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


class FaceAligner:
    def __init__(self, desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")
        #self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    #def align(self, image, gray, rect=None, lms=None):
    def align(self, image, shape=None):
        # convert the landmark (x, y)-coordinates to a NumPy array
        #if lms is None:
        #    shape = self.predictor(gray, rect)
        #    shape = shape_to_np(shape)
        #elif rect is None:
        #    shape = np.asarray(lms)
        
        #print(type(shape), len(shape))
        #print(len(shape), shape)
        #simple hack ;)
        
        if shape is None:
            print('########## Shape is None! ##########')
            dets = self.detector(image, 1)
            if len(dets) == 0:
                print('########## No detections! ##########')
                return image
            shape = self.predictor(image, dets[0])
            shape = shape_to_np(shape)
        
        if (len(shape)==68):
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]
            
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        #desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        #dist = np.sqrt((dX ** 2) + (dY ** 2))
        #desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        #desiredDist *= self.desiredFaceWidth
        #scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2, (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale=1.0)

        # update the translation component of the matrix
        #tX = self.desiredFaceWidth * 0.5
        #tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        #M[0, 2] += (tX - eyesCenter[0])
        #M[1, 2] += (tY - eyesCenter[1])
        
        offset = np.zeros(M.shape[1], dtype=M.dtype)
        offset[:2] = M[:,2]
        offset[2] = 0
        
        matrix = np.zeros((M.shape[0]+1, M.shape[0]+1), dtype=M.dtype)
        matrix[:2,:2] = M[:,:2]
        matrix[2,2] = 1
        
        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        
        #output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        output = scipy.ndimage.affine_transform(image, matrix, offset, mode='reflect')
        
        # return the aligned face
        return output

    
class FaceDetector(ABC):

    @abstractmethod
    def run(self, image, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)


class FAN(FaceDetector):

    def __init__(self, device = 'cuda', threshold=0.5):
        self.face_detector = 'sfd'
        self.face_detector_kwargs = {
            "filter_threshold": threshold
        }
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                                  device=str(device),
                                                  flip_input=False,
                                                  face_detector=self.face_detector,
                                                  face_detector_kwargs=self.face_detector_kwargs)

    # @profile
    def run(self, image, with_landmarks=False, detected_faces=None):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image, detected_faces=detected_faces)
        torch.cuda.empty_cache()
        if out is None:
            del out
            if with_landmarks:
                return [], 'kpt68', []
            else:
                return [], 'kpt68'
        else:
            boxes = []
            kpts = []
            for i in range(len(out)):
                kpt = out[i].squeeze()
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                bbox = [left, top, right, bottom]
                boxes += [bbox]
                kpts += [kpt]
            del out # attempt to prevent memory leaks
            if with_landmarks:
                return boxes, 'kpt68', kpts
            else:
                return boxes, 'kpt68'
            
            
def lms_to_np(lms):
    lms = [int(float(l)) for l in lms.split(';')]
    #x_cor_list, y_cor_list = [], []
    #for i in range(len(lms)):
    #    if i % 2 == 0:
    #        x_cor_list.append(float(lms[i]))
    #    else:
    #        y_cor_list.append(float(lms[i]))
    #lms = [x_cor_list, y_cor_list]
    #lms = np.asarray(lms)  # [2, 68]
    lms = list(zip(*(iter(lms),) * 2))
    return lms


def crop_align_affectnet():
    img_root = '/gpu-data2/panto/affectnet_manually'
#     save_root = os.path.join('/gpu-data2/jpik/affectnet/', 'val_aligned_v3')
    save_root = os.path.join('/gpu-data2/jpik/affectnet/', 'train_aligned_v3')
    
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    #detector = dlib.cnn_face_detection_model_v1("assets/mmod_human_face_detector.dat")
    #detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")
    #my_fa = MyFaceAligner(desiredLeftEye=(0.3, 0.3), desiredFaceWidth=256)
    
    
    IMAGE_SIZE = 224
    fa = FaceAligner(desiredFaceWidth=IMAGE_SIZE, desiredFaceHeight=None)
    face_detector = FAN()
  
    cnt = 0
    row_cnt = 1
    with open('/gpu-data2/panto/affectnet_manually/Manually_Annotated_file_lists/training.csv', 'r') as csvfile:
    #with open('/gpu-data2/panto/affectnet_manually/Manually_Annotated_file_lists/validation.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cur_sample = {}
            cur_sample['img_path'] = os.path.join(img_root, row['subDirectory_filePath'])
            #lms = row['facial_landmarks']
            #cur_sample['lms'] = lms_to_np(lms)
            #print(cur_sample['lms'])
            #print(len(cur_sample['lms']))
            cur_sample['expression'] = int(row['expression'][0:])
            x, y, w, h = int(row['face_x']), int(row['face_y']), int(row['face_width']), int(row['face_height'])
            assert x == y and w == h
            # affectnet emotion label:
            # 0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt
            # 8: None, 9: Uncertain, 10: No-Face
            if 0 <= cur_sample['expression'] <= 7:
                print(f"Counter: {cnt}, Row: {row_cnt}, Path: {cur_sample['img_path']}")
                row_cnt += 1
                #img = cv2.imread(cur_sample['img_path'])
                #img = np.array(Image.open(cur_sample['img_path']))
                
                img_name = row['subDirectory_filePath'].split('/')[1]
                save_path = os.path.join(save_root, f"{cur_sample['expression']}_{img_name}") 
                if os.path.exists(save_path):
                    print(f"Image {save_path} already exists!")
                    continue
                img = np.array(imread(cur_sample['img_path']))
                # img = img[x:x+w,y:y+h,:] # Crop using precomputed boxes
                img_ = (resize(img, (256, 256))*255).astype(np.uint8) # resize to desired size
                bbox, bbox_type, landmarks = face_detector.run(img_, with_landmarks=True) # Find landmarks    
                
                if len(bbox) > 0:
                    x1, y1, x2, y2 = [max(int(d),0) for d in bbox[0]]
                    offset = (max(x2-x1, y2-y1) - min(x2-x1, y2-y1)) // 2
                    if x2-x1 < y2-y1:
                        x1 = max(x1-offset,0)
                        img_ = img_[y1:y2,x1:x2+offset]
                    elif x2-x1 > y2-y1:
                        y1 = max(y1-offset,0)
                        img_ = img_[y1:y2+offset,x1:x2]
                    else:
                        img_ = img_[y1:y2,x1:x2]
                else:
                    img_ = img[x:x+w,y:y+h,:]    
                
                img_ = (resize(img_, (IMAGE_SIZE, IMAGE_SIZE))*255).astype(np.uint8)                 
                bbox, bbox_type, landmarks = face_detector.run(img_, with_landmarks=True) # Find landmarks    
          
                # use 68 lms provided by AffectNet
                #result = my_fa.align(img, cur_sample['lms'])      
                #dets = detector(img, 1)
                #if dets is not None:
                if len(landmarks) > 0:
                    aligned_img = fa.align(img_, shape=np.asarray(landmarks[0]))
                    #print(len(dets))
                    #if len(dets) > 0:
                        #aligned_img = fa.align(img, img, rect=dets[0].rect, lms=None)
                        #aligned_img = fa.align(img, img, rect=dets[0], lms=None)
                    #else:
                    #    aligned_img = fa.align(img, img, rect=None, lms=cur_sample['lms'])
                else:    
                    aligned_img = fa.align(img_, shape=None)
                
                #cv2.imwrite(save_path, result)
                #aligned_img = Image.fromarray(aligned_img)
                #aligned_img.save(save_path)
                imsave(save_path, aligned_img)
                cnt += 1
            
#             if cnt > 20: break;

    print('# of saved images:', cnt)
    
    
if __name__ == '__main__':
    crop_align_affectnet()