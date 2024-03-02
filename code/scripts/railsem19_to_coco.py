from os import listdir
from os.path import isfile, join
import cv2
import json
import shutil
import os

input_dataset_path = "D:/Mano/Rail/RailSem19/"
output_dataset_path = "D:/Mano/coco/"

DESIRED_CLASSES = ["switch-left", "switch-right"]

def get_license():
    licenses = {}
    licenses['0'] = {'url': "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                     'id': 1,
                     'name': "Attribution-NonCommercial-ShareAlike License"}
    
    licenses['1'] = {'url': "http://creativecommons.org/licenses/by-nc/2.0/",
                     'id': 2,
                     'name': "Attribution-NonCommercial License"}
    
    licenses['2'] = {'url': "http://creativecommons.org/licenses/by-nc-nd/2.0/",
                     'id': 3,
                     'name': "Attribution-NonCommercial-NoDerivs License"}
    
    licenses['3'] = {'url': "http://creativecommons.org/licenses/by/2.0/",
                     'id': 4,
                     'name': "Attribution License"}
    
    licenses['4'] = {'url': "http://creativecommons.org/licenses/by-sa/2.0/",
                     'id': 5,
                     'name': "Attribution-ShareAlike License"}
    
    licenses['5'] = {'url': "http://creativecommons.org/licenses/by-nd/2.0/",
                     'id': 6,
                     'name': "Attribution-NoDerivs License"}
    
    licenses['6'] = {'url': "http://flickr.com/commons/usage/",
                     'id': 7,
                     'name': "No known copyright restrictions"}
    
    licenses['7'] = {'url': "http://www.usa.gov/copyright.shtml",
                     'id': 8,
                     'name': "United States Government Work"}
    
    return licenses


def get_files_from_folder(path):

    return [f for f in listdir(path) if isfile(join(path, f))]

def convert_railsem_to_coco():

    image_path = input_dataset_path + 'jpgs/rs19_val/'
    bbox_path = input_dataset_path + 'jsons/rs19_val/'
    image_files = get_files_from_folder(image_path)
    bbox_files = get_files_from_folder(bbox_path)

    instances_data = {}
    # INFO
    info = {}
    info['description'] = "RailSem19"
    info['url'] = "https://www.wilddash.cc/railsem19"
    info['version'] = "1.0"
    info['year'] = 2019
    info['contributor'] = "Zendel et al."
    info['date_created'] = "2019"
    instances_data['info'] = info

    # LICENSE
    instances_data['licenses'] = get_license()

    # IMAGES
    images = {}
    no_images = len(image_files)
    if not os.path.isdir(output_dataset_path + 'images/'):
        os.mkdir(output_dataset_path + 'images/')
    if not os.path.isdir(output_dataset_path + 'images/train2017/'):
        os.mkdir(output_dataset_path + 'images/train2017/')
    if not os.path.isdir(output_dataset_path + 'images/val2017/'):
        os.mkdir(output_dataset_path + 'images/val2017/')
    if not os.path.isdir(output_dataset_path + 'images/test2017/'):
        os.mkdir(output_dataset_path + 'images/test2017/')

    images = {'train2017':[], 'val2017':[], 'test2017':[]}
    for im_id, im in enumerate(image_files):

        img = cv2.imread(image_path + im, cv2.IMREAD_UNCHANGED)
        height, width = img.shape[:2]

        '''
        images[im_id] = {
            'license': 7,
            'file_name': im,
            'coco_url' : "404 Not Found",
            'height': height,
            'width': width,
            'date_captured': "2019",
            'flickr_url': "404 Not Found",
            'id': im_id
        }
        '''
        image = {
            'license': 7,
            'file_name': im,
            'coco_url' : "404 Not Found",
            'height': height,
            'width': width,
            'date_captured': "2019",
            'flickr_url': "404 Not Found",
            'id': im_id
        }

        if im_id < no_images * 0.7:
            subset = 'train2017'
        elif im_id < no_images * 0.9:
            subset = 'val2017'
        else:
            subset = 'test2017'

        images[subset].append(image)
        
        shutil.copy(image_path + im, output_dataset_path + 'images/' + subset + '/' + im)

    # instances_data['images'] = images

    # ANNOTATIONS
    annotations = {'train2017':[], 'val2017':[], 'test2017':[]}
    bf_id = -1
    for im_id, bf in enumerate(bbox_files):
        f = open(bbox_path + bf)
        data = json.load(f)
        for bbox in data['objects']:
            if bbox['label'] in DESIRED_CLASSES:
                class_no = DESIRED_CLASSES.index(bbox['label'])
                railsem_bbox = bbox['boundingbox']
                x1, y1, x2, y2 = railsem_bbox[0], railsem_bbox[1], railsem_bbox[2], railsem_bbox[3]
                bf_id += 1
                annotation = {
                    'image_id' : im_id,
                    'iscrowd': 0,
                    'area': (x2-x1) * (y2-y1),
                    'bbox': [x1, y1, (x2-x1), (y2-y1)],
                    'segmentation' : [[x1,y1,x1,(y1 + y2), (x1 + x2), (y1 + y2), (x1 + x2), y1]],
                    'category_id': class_no,
                    'id': bf_id
                }

                if im_id < no_images * 0.7:
                    subset = 'train2017'
                elif im_id < no_images * 0.9:
                    subset = 'val2017'
                else:
                    subset = 'test2017'

                annotations[subset].append(annotation)
            
    # instances_data['annotations'] = annotations
    
    instances_data['categories'] = [{"supercategory": "switch", "id": 0, "name": "switch-left"}, 
                                    {"supercategory": "switch", "id": 1, "name": "switch-right"}]
    
    train_json_data = {'info' : instances_data['info'], 'licenses':instances_data['licenses'], 'images':images['train2017'], 'annotations':annotations['train2017'], 'categories':instances_data['categories']}
    valid_json_data = {'info' : instances_data['info'], 'licenses':instances_data['licenses'], 'images':images['val2017'], 'annotations':annotations['val2017'], 'categories':instances_data['categories']}
    test_json_data = {'info' : instances_data['info'], 'licenses':instances_data['licenses'], 'images':images['test2017'], 'annotations':annotations['test2017'], 'categories':instances_data['categories']}
    
    if not os.path.isdir(output_dataset_path + 'annotations/'):
        os.mkdir(output_dataset_path + 'annotations/')
    with open(output_dataset_path + 'annotations/instances_train2017.json', 'w') as f:
        json.dump(train_json_data, f, indent=4)
    with open(output_dataset_path + 'annotations/instances_val2017.json', 'w') as f:
        json.dump(valid_json_data, f, indent=4)
    with open(output_dataset_path + 'annotations/instances_test2017.json', 'w') as f:
        json.dump(test_json_data, f, indent=4)
    
convert_railsem_to_coco()