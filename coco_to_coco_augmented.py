import os
import torch
import torch.utils.data
import torchvision
import numpy as np
import cv2
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms
from bbaug.policies import policies
from shutil import copyfile
import json

from torchvision import transforms

class cocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, policy_container=None, just_aug=None):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.policy_container = policy_container
        self.to_tensor = transforms.ToTensor()
        self.tensor_to_image = transforms.ToPILImage()
        self.just_aug = just_aug

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # image name for input image
        imgName = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = np.array(Image.open(os.path.join(self.root, imgName)))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        boxes_orig=[]
        for i in range(num_objs):
            label = coco_annotation[i]['category_id']
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(label))
            boxes_orig.append([xmin, ymin, xmax, ymax,label])

        if self.policy_container:

            # Select a random sub-policy from the policy list
            random_policy = self.policy_container.select_random_policy()
            #print(random_policy)

            # Apply this augmentation to the image, returns the augmented image and bounding boxes
            # The boxes must be at a pixel level. e.g. x_min, y_min, x_max, y_max with pixel values
            img_aug, bbs_aug = self.policy_container.apply_augmentation(
                random_policy,
                img,
                boxes,
                labels,
            )
            bbs_aug_formated = []
            for label,xmin, ymin, xmax, ymax in bbs_aug:
                bbs_aug_formated.append([xmin, ymin, xmax, ymax,label])
            
            # Only return the augmented image, bounded boxes if there are
            # boxes present after the image augmentation and the image name without extension filename
            
            #---------------------------------------------
            '''img = self.to_tensor(img)
            img = np.asarray(self.tensor_to_image(img))
            img_aug = self.to_tensor(img_aug)
            img_aug = np.asarray(self.tensor_to_image(img))'''
            if bbs_aug.size > 0:
                    return img, boxes_orig, img_aug, bbs_aug_formated ,imgName
            else:
                return img, boxes_orig, [], np.array([]), imgName
        return img, boxes_orig, imgName

    def collate_fn(self, batch):
        """
        Custom collate function to incorporate the augmentations into the 
        input tensor
        """
        if self.policy_container:
            if self.just_aug:
                imgs, targets, imgs_aug, targets_aug, imgs_names = list(zip(*batch))
                imgs_names_list = []
                for idx in range(len(imgs)):
                    img_name = imgs_names[idx][0:len(imgs_names[idx])-4]
                    file_format = imgs_names[idx][len(imgs_names[idx])-4:]
                    imgs_names_list.append(img_name+'_aug'+file_format)
                return imgs_aug, targets_aug, imgs_names_list
            else:
                #---------------------------------------------
                imgs, targets, imgs_aug, targets_aug, imgs_names = list(zip(*batch))
                #---------------------------------------------
                images_list = []
                annots_list = []
                imgs_names_list = []
                for idx in range(len(imgs)):
                    img_name = imgs_names[idx][0:len(imgs_names[idx])-4]
                    file_format = imgs_names[idx][len(imgs_names[idx])-4:]

                    images_list.append(imgs[idx])
                    annots_list.append(targets[idx])
                    imgs_names_list.append(img_name+file_format)

                    images_list.append(imgs_aug[idx])
                    annots_list.append(targets_aug[idx])
                    imgs_names_list.append(img_name+'_aug'+file_format)

                #---------------------------------------------
                return images_list, annots_list, imgs_names_list
                #---------------------------------------------
    
    def __len__(self):
        return len(self.ids)


def lists_to_json(class_list):
    """
    Transform the class list into json.

    Params
    :class_file(list of classes) string list of the names of the classes.

    Returns
    :anns_list(json) correct format of json as defined in coco.
    """
    json_data = {}
    #print(annotation_array)
    
    #main nodes
    json_data['info'] = {}
    json_data['licenses'] = []
    json_data['categories'] = []
    json_data['images'] = []
    json_data['annotations'] = []

    #license, not used but included
    json_data['info']['year'] = "2021"
    json_data['info']['version'] = "11"
    json_data['info']['description'] = ""
    json_data['info']['contributor'] = ""
    json_data['info']['url'] = ""
    json_data['info']['date_created'] = "2041-06-16T14:01:38+00:00"
    
    json_data['licenses'].append({
        'id': 1,
        'url': '',
        'name': 'Unknown'
    })
    
    #categories the id will be the position in the array
    for i, item in enumerate(class_list):
        json_data['categories'].append({
            'id': i+1,
            'name': item,
            'supercategory': 'none'
        })
        
    #images and annotation should be included later on
    
    return json_data
def annotations_to_json(im, img_name, img_index, img_extension, 
                        ann_counter, anns_names_list, anns_bboxes_list, json_data):
    #get width and height
    h, w, c = im.shape

    json_data['images'].append({
        'id': img_index,
        'license': 1,
        'file_name': img_name + '.' + img_extension,
        'height': h,
        'width': w,
        'date_captured': '2041-03-03T03:54:32+00:00'

    })

    #get annotations of this image
    annotation_index = anns_names_list.index(img_name + '.' + img_extension)
    for annotation in anns_bboxes_list[annotation_index]:
        x1,y1,x2,y2,class_id = annotation
        x1,y1,x2,y2,class_id = float(x1), float(y1), float(x2), float(y2), int(class_id)
        
        json_data['annotations'].append({
            'id': ann_counter,
            'image_id': img_index,
            'category_id': class_id,
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'area': (x2 - x1) * (y2 - y1),
            'segmentation': [],
            'iscrowd': 0
        })
        ann_counter += 1
    return json_data, ann_counter

def generate_COCO_Dataset_transformed(output_imgs_folder,
                                      output_json_name,
                                      obj_list,
                                      inpur_dir_images,
                                      input_coco_anns,
                                      aug_policy,
                                      num_of_workers,
                                      batch_size):
    '''
     Generate a transformation of the ground truth with the specified augmentation policy

    Params
    :output_imgs_folder (str) -> path where the transformed images will be stored.
    :output_json_name (str) -> path and name where the new json file will be stored.
    :obj_list (list<int>) -> classes of the annotations.
    :inpur_dir_images (pycocotools.coco) -> current path of the images
    :input_coco_anns (EfficientDetBackbone) -> current path and name of the json file
    :aug_policy List<POLICY_TUPLE_TYPE> -> policy to apply.
    :num_of_workers (int) -> number of workers for the dataloader of the dataset
    :batch_size (int) -> size of the batch for the dataloader of the dataset
    Return
    :annotations file (pycocotools.coco)
    :transformed images (.jpg)
    '''
    tensor_to_image = transforms.ToPILImage()

    aug_policy_container = policies.PolicyContainer(aug_policy, random_state=42)
    # Generate the dataloader just with transformed images.
    write_just_aug = True
    # select device (whether GPU or CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    # create own Dataset
    if write_just_aug:
        my_dataset = cocoDataset(root=inpur_dir_images,
                                annotation=input_coco_anns, 
                                policy_container=aug_policy_container,
                                just_aug=True
                                )
    else:
        my_dataset = cocoDataset(root=inpur_dir_images,
                                annotation=input_coco_anns, 
                                policy_container=aug_policy_container
                                )
    # own DataLoader
    data_extractor = torch.utils.data.DataLoader(my_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_of_workers,
                                            collate_fn=my_dataset.collate_fn)

    json_data_train = lists_to_json(obj_list)
    image_id = 0
    ann_counter = 0
    for batch in data_extractor:
        images, targets, imgs_names = batch

        for img_index, img_name in enumerate(imgs_names):
            img_extension = img_name[len(img_name)-3:]
            img_name =  img_name[0:len(img_name)-4]   
            img = images[img_index]

            tensor_to_image(img).save(output_imgs_folder +img_name+'.'+img_extension)

            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #cv2.imwrite(output_imgs_folder +img_name+'.'+img_extension,image)
            
            #include annotations in the json
            final_json_train, ann_counter = annotations_to_json(img, img_name, image_id, img_extension, ann_counter,
                                                                        imgs_names, targets, json_data_train)

            #save json file
            with open(output_json_name, 'w') as outfile:
                json.dump(final_json_train, outfile, indent=4)
            image_id = image_id + 1
        #-------------------
# -------------------------------------------------------------------------------#