from PIL import Image
import torchvision.transforms.functional as TF
import random
import numpy as np
from skimage.util import random_noise
from skimage.exposure.exposure import rescale_intensity
from skimage.draw import polygon
import cv2

class Get_ROI(object):
    def __init__(self, keep_size=12,pad_flag=False):
        self.keep_size = keep_size
        self.pad_flag = pad_flag
    
    def __call__(self, image):
        '''
        sample['image'] must be scaled to (0~1)
        '''
        h,w = image.shape
        roi = self.get_body(image)

        if np.sum(roi) != 0:
            roi_nz = np.nonzero(roi)
            roi_bbox = [
                np.maximum((np.amin(roi_nz[0]) - self.keep_size), 0), # left_top x
                np.maximum((np.amin(roi_nz[1]) - self.keep_size), 0), # left_top y
                np.minimum((np.amax(roi_nz[0]) + self.keep_size), h), # right_bottom x
                np.minimum((np.amax(roi_nz[1]) + self.keep_size), w)  # right_bottom y
            ]
        else:
            roi_bbox = [0,0,h,w]
        
        image = image[roi_bbox[0]:roi_bbox[2],roi_bbox[1]:roi_bbox[3]]
        # pad
        if self.pad_flag:
            nh, nw = roi_bbox[2] - roi_bbox[0], roi_bbox[3] - roi_bbox[1]
            if abs(nh - nw) > 1:
                if nh > nw:
                    pad = ((0,0),(int(nh-nw)//2,int(nh-nw)//2))
                else:
                    pad = ((int(nw-nh)//2,int(nw-nh)//2),(0,0))
                image = np.pad(image,pad,'constant')

        return image
    
    def get_body(self,image):
        body_array = np.zeros_like(image, dtype=np.uint8)
        img = rescale_intensity(image, out_range=(0, 255))
        img = img.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        body = cv2.erode(img, kernel, iterations=1)
        blur = cv2.GaussianBlur(body, (5, 5), 0)
        _, body = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        area = [[c, cv2.contourArea(contours[c])] for c in range(len(contours))]
        area.sort(key=lambda x: x[1], reverse=True)
        body = np.zeros_like(body, dtype=np.uint8)
        for j in range(min(len(area),3)):
            if area[j][1] > area[0][1] / 20:
                contour = contours[area[j][0]]
                r = contour[:, 0, 1]
                c = contour[:, 0, 0]
                rr, cc = polygon(r, c)
                body[rr, cc] = 1
        body_array = cv2.medianBlur(body, 5)

        return body_array


class Trunc_and_Normalize(object):
  '''
  truncate gray scale and normalize to [0,1]
  '''
  def __init__(self, scale):
    self.scale = scale
    assert len(self.scale) == 2, 'scale error'

  def __call__(self, image):
 
      # gray truncation
      image = image - self.scale[0]
      gray_range = self.scale[1] - self.scale[0]
      image[image < 0] = 0
      image[image > gray_range] = gray_range
      
      # normalization
      # image = (image - np.min(image)) / (np.max(image) - np.min(image))
      image = image / gray_range

      return image


class Convert2PIL(object):
    """
    Convert numpy to PIL image
    """

    def __init__(self, channels=1):
        self.channels = channels

    def __call__(self, img):
        if self.channels == 1:
            image = Image.fromarray(img*255).convert('L')
        elif self.channels == 3:
            image = Image.fromarray(img*255).convert('RGB')

        return image


class RandomRotate(object):

    def __init__(self, angels):
        self.angels = angels

    def __call__(self, image):

        angle = random.choice(self.angels)
        image = TF.rotate(image, angle)
        return image

class AddNoise(object):

    def __call__(self, image):
        if image.mode == 'RGB':
            image = random_noise(np.array(image)[...,0],mode='s&p') 
            image = Image.fromarray((image*255).astype(np.uint8)).convert('RGB')
        else:
            image = random_noise(np.array(image),mode='s&p') 
            image = Image.fromarray((image*255).astype(np.uint8)).convert('L')
        return image
