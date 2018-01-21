import os
from os import listdir
from os.path import isfile, join
import numpy as np
import scipy as sp
import xmltodict
from scipy import misc
import pickle

ROOT_PATH = "VOCtrainval_11-May-2012/VOCdevkit/VOC2012"
ANNOTATION_FOLDER = "Annotations"
IMAGE_FOLDER = "JPEGImages"
CLASSES = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car",
           "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]

CLASS_TO_LABEL = {class_name: label for (class_name, label) in zip(CLASSES, range(len(CLASSES)))}

def extract_objects(xml):
    objects = xml['annotation']['object']
    return [objects] if isinstance(objects,xmltodict.OrderedDict) else objects

def extract_labels(xml):
    return [CLASS_TO_LABEL[object['name']] for object in extract_objects(xml)]


def extract_bounding_boxes(xml):
    return [tuple([int(round(float(object['bndbox'][key])))
                   for key in ['xmin', 'ymin', 'xmax', 'ymax']]) for object in extract_objects(xml)]


def extract_subimages(image):
    
    (w,h,d) = image.shape
    (new_w,new_h) = (256,(h*256)//w) if w < h else ((w*256)//h,256)

    image = sp.misc.imresize(image, (new_w,new_h,d))
    (w,h,d) = image.shape
    image = image[w//2-128:w//2+128,h//2-128:h//2+128,:]
           
    new_w = 256
    new_h = 256
    
    crops = [
        #nw
        image[0:225,0:225,:],
        #ne
        image[0:225,new_h-225:new_h,:],
        #sw
        image[new_w-225:new_w,0:225,:],
        #se
        image[new_w-225:new_w,new_h-225:new_h,:],
        #c
        image[new_w//2-112:new_w//2+113,new_h//2-112:new_h//2+113,:],
    ]
           
    crop_flips = [np.fliplr(crop) for crop in crops]
    subimages = crops
    subimages.extend(crop_flips)
      
    return subimages



def format_data(data):
    
    values = [
        [[subimage,value["labels"]]
         for subimage in value["subimages"]]
              for key,value in data.items()]
    
    #flatten values
    values = [item for sublist in values for item in sublist]
    
    return values


def process_data():
           
    annotation_directory_path = os.path.join(ROOT_PATH, ANNOTATION_FOLDER)
    image_directory_path = os.path.join(ROOT_PATH, IMAGE_FOLDER)
           
    names_and_image_paths_and_xml_paths = [
        (os.path.splitext(filename)[0],join(image_directory_path, os.path.splitext(filename)[0] + ".jpg"), join(annotation_directory_path, filename))
        for filename in listdir(annotation_directory_path)]

    names_and_image_paths_and_xml_paths = [(name,image_path, annotation_path)
                                           for (name,image_path, annotation_path) in names_and_image_paths_and_xml_paths
                                           if isfile(annotation_path)]
           
    data = {}

    for (name,image_path, annotation_path) in names_and_image_paths_and_xml_paths:
        xml = xmltodict.parse(open(annotation_path, 'rb'))

        image = misc.imread(image_path, mode='RGB')
        subimages = extract_subimages(image)
        labels = extract_labels(xml)
        bounding_boxes = extract_bounding_boxes(xml)
        
        data[name] = {
            "image":image,
            "subimages":subimages,
            "labels":labels,
            "bounding_boxes":bounding_boxes
        }

        print("Processed %s" % name)
    
    
    formatted_data = format_data(data)

    for ((image,label),i) in zip(formatted_data,range(len(formatted_data))):
        sp.misc.imsave("out/"+str(i)+".png",image)
        print(str(i))
           
    labels = [label for (image,label) in formatted_data]
    pickle.dump(labels, open("data.p", "wb"),protocol=4)
    

process_data()


