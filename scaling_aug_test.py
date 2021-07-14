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

aug_policy = policies.policies_pineapples_zoom_in_5()
aug_policy_container = policies.PolicyContainer(aug_policy)

class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, policy_container=None):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.policy_container = policy_container
        self.to_tensor = transforms.ToTensor()

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
        for i in range(num_objs):
            label = coco_annotation[i]['category_id']
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(label))

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
            labels = np.array(labels)
            img = self.to_tensor(img) # Convert the image to a tensor
            boxes = np.hstack((np.vstack(labels), np.array(boxes))) # Add the labels to the boxes
            img_aug = self.to_tensor(img_aug) # Convert the augmented image to a tensor
            bbs_aug= np.array(bbs_aug)
            
            # Only return the augmented image, bounded boxes if there are
            # boxes present after the image augmentation and the image name without extension filename
            
            #---------------------------------------------
            if bbs_aug.size > 0:
                return img, boxes, img_aug, bbs_aug ,imgName
            else:
                return img, boxes, [], np.array([]), imgName
        return img, boxes, imgName
        '''#boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        #labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, my_annotation'''

    def collate_fn(self, batch):
        """
        Custom collate function to incorporate the augmentations into the 
        input tensor
        """
        if self.policy_container:
            #---------------------------------------------
            imgs, targets, imgs_aug, targets_aug, imgs_names = list(zip(*batch))
            #---------------------------------------------
            #print(type(imgs_aug))
            #print(type(targets_aug))
            # Create the image and target list for the unaugmented data
            imgs = [i for i in imgs]
            targets = [i for i in targets]
            
            #---------------------------------------------
            imgs_names = [i for i in imgs_names]
            #---------------------------------------------
            
            # Only add the augmented images and targets if there are targets
            for i, box_aug in enumerate(targets_aug):
                if box_aug.size > 0:
                    imgs.append(imgs_aug[i])
                    targets.append(box_aug)

            # Stack the unaugmented and augmented images together
            imgs = torch.stack(imgs)
            
            # Concatenate the unaugmented and augmented targets together
            # also add the sample index to the first column
            for i in range(len(imgs)):
                targets[i] = torch.Tensor(np.insert(targets[i], 0, i, axis=1))
            targets = torch.cat(targets, 0)
            
            #---------------------------------------------
            return imgs, imgs_names, targets
            #---------------------------------------------
    
    def __len__(self):
        return len(self.ids)

def getAllBboxesFromSpecificImg(img_idx, targets):
    annotations = []
    for target in targets:
        if target[0] == img_idx:
            annotations.append([int(target[1].item()),int(target[2].item()),int(target[3].item()),int(target[4].item()),int(target[5].item())])
        
    return annotations
#path to your own data and coco file
train_data_dir = 'D:/Manfred/InvestigacionPinas/Beca-CENAT/workspace/5m_train_valid_test_vl/test'
train_coco = 'D:/Manfred/InvestigacionPinas/Beca-CENAT/workspace/5m_train_valid_test_vl/annotations/instances_test.json'

# create own Dataset
my_dataset = myOwnDataset(root=train_data_dir,
                          annotation=train_coco, 
                          policy_container=aug_policy_container
                          )
# Batch size
train_batch_size = 3

# own DataLoader
data_loader = torch.utils.data.DataLoader(my_dataset,
                                          batch_size=train_batch_size,
                                          shuffle=True,
                                          num_workers=0,
                                          collate_fn=my_dataset.collate_fn)

# select device (whether GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# DataLoader is iterable over Dataset
'''for batch in data_loader:
    images, imgs_names, targets = batch
    annotations = getAllBboxesFromSpecificImg(0, targets)
    print(images.size())
    print(len(annotations))
    #print(annotations)'''

tensor_to_image = transforms.ToPILImage()
def printROIS(labelsTensor,imageToPrint):
  for idx,label,x1,y1,x2,y2 in labelsTensor:
    left_top = (int(x1.item()), int(y1.item()))
    right_bottom = (int(x2.item()), int(y2.item()))
    cv2.rectangle(imageToPrint, left_top, right_bottom, (255, 0, 0), 2)
  return imageToPrint

images, imgs_names, targets = next(iter(data_loader))
imgs_names = imgs_names + imgs_names
for idx in range(images.size()[0]):
    image_numpy = np.asarray(tensor_to_image(images[idx]))
    annotations = targets[targets[:,0]==idx]
    if idx<train_batch_size:
        cv2.imwrite(f'D:/Manfred/InvestigacionPinas/Beca-CENAT/workspace/prueba_dataaugmentation/{imgs_names[idx]}', printROIS(annotations,image_numpy))
    else:
        img_name = imgs_names[idx][0:len(imgs_names[idx])-4]
        cv2.imwrite(f'D:/Manfred/InvestigacionPinas/Beca-CENAT/workspace/prueba_dataaugmentation/{img_name}_aug.JPG', printROIS(annotations,image_numpy))

