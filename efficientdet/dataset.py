import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from torchvision import transforms
import cv2


class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None, policy_container=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()
        #-----------------
        self.policy_container = policy_container
        self.to_tensor = transforms.ToTensor()
        #-----------------

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, image_index):

        #img = self.load_image(idx)
        imgName = self.coco.loadImgs(self.image_ids[image_index])[0]['file_name']
        img = cv2.imread(os.path.join(self.root_dir, self.set_name, imgName))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = img.astype(np.float32) / 255.  ### WHAT ABOUT THIS? *******************


        #-----------------------------------
        # List: get annotation id from coco
        ann_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)

        # Dictionary: target coco_annotation file for an image
        coco_annotation = self.coco.loadAnns(ann_ids)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in range(len(coco_annotation)):
            label = coco_annotation[i]['category_id'] - 1
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(label))
        

        #sample = {'img': img, 'annot': annot}   ### WHAT ABOUT THIS? *******************
        

        if self.policy_container:
            # Select a random sub-policy from the policy list
            random_policy = self.policy_container.select_random_policy()

            # Apply this augmentation to the image, returns the augmented image and bounding boxes
            # The boxes must be at a pixel level. e.g. x_min, y_min, x_max, y_max with pixel values
            img_aug, bbs_aug = self.policy_container.apply_augmentation(
                random_policy,
                img,
                boxes,
                labels,
            )

            labels = np.array(labels)
            #img = self.to_tensor(img) # Convert the image to a tensor
            boxes = np.hstack(( np.array(boxes), np.vstack(labels) )) # Add the labels to the boxes
            #img_aug = self.to_tensor(img_aug) # Convert the augmented image to a tensor
            bbs_aug = np.array(bbs_aug)
            bbs_aug = bbs_aug[:, [1,2,3,4,0]].astype(np.float32) # send category to the end of the row 

            sample_original = {'img': img, 'annot': boxes}
            sample_augmented = {'img': img_aug, 'annot': bbs_aug}
            if self.transform:
                sample_original = self.transform(sample_original)
                sample_augmented = self.transform(sample_augmented)

            img_t, boxes_t = sample_original['img'], sample_original['annot']
            img_aug_t, bbs_aug_t = sample_augmented['img'], sample_augmented['annot']

            # Only return the augmented image, bounded boxes if there are
            # boxes present after the image augmentation and the image name without extension filename
            #---------------------------------------------
            if bbs_aug_t.numel() > 0:
                return img_t, boxes_t.squeeze(), imgName, img_aug_t, bbs_aug_t.squeeze(), "aug_" + imgName
            else:
                return img_t, boxes_t.squeeze(), imgName, torch.tensor([]), torch.tensor([]), ""


        # No augmentation
        else:
            boxes = np.hstack(( np.array(boxes), np.vstack(labels) ))
            sample = {'img': img, 'annot': boxes}
            if self.transform:
                sample = self.transform(sample)
            img_, boxes_ = sample['img'], sample['annot']
            return img_, boxes_.squeeze(), imgName
        

    def collater(self, batch):

        if self.policy_container:
            #---------------------------------------------
            imgs, annots, imgs_names, imgs_aug, annots_aug, imgs_names_aug = list(zip(*batch))
            #---------------------------------------------

            # Create the image and target list for the unaugmented data
            imgs = [i for i in imgs]
            annots = [i for i in annots]
            imgs_names = [i for i in imgs_names]
            
            # Only add the augmented images and annots if there are annots
            for i, box_aug in enumerate(annots_aug):
                if box_aug.numel() > 0:
                    imgs.append(imgs_aug[i])
                    annots.append(box_aug)
                    imgs_names.append(imgs_names_aug[i])

            # Stack the unaugmented and augmented images together
            imgs = torch.stack(imgs)

            # Add padding
            annots = [an if len(list(an.shape)) != 1 else an.unsqueeze(dim=0) for an in annots]
            max_num_annots = max(annot.shape[0] for annot in annots) 
            if max_num_annots > 0:
                annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
                for idx, annot in enumerate(annots):
                    if annot.shape[0] > 0:
                        annot_padded[idx, :annot.shape[0], :] = annot
            else:
                annot_padded = torch.ones((len(annots), 1, 5)) * -1

            return {'img': imgs, 'annot': annot_padded, 'img_names': imgs_names}


        # No augmentation
        else:
            imgs, annots, imgs_names = list(zip(*batch))

            imgs = [i for i in imgs]
            annots = [i for i in annots]
            imgs_names = [i for i in imgs_names]

            annots = [an if len(list(an.shape)) != 1 else an.unsqueeze(dim=0) for an in annots]
            max_num_annots = max(annot.shape[0] for annot in annots)

            if max_num_annots > 0:
                annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

                for idx, annot in enumerate(annots):
                    if annot.shape[0] > 0:
                        annot_padded[idx, :annot.shape[0], :] = annot
            else:
                annot_padded = torch.ones((len(annots), 1, 5)) * -1

            imgs = torch.stack(imgs)
            return {'img': imgs, 'annot': annot_padded, 'img_names': imgs_names}
        


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        #return {'img': self.to_tensor(new_image).to(torch.float32), 'annot': self.to_tensor(annots), 'scale': scale}
        return {'img': self.to_tensor(new_image).to(torch.float32), 'annot': self.to_tensor(annots)}

'''
class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample
'''


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
