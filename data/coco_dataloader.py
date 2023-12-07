import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torch.utils.data import DataLoader, random_split

from pycocotools.coco import COCO
from pycocotools import mask as coco_mask


# Change the working directory to the location of this script
Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
os.chdir(Path)


class COCODataset(Dataset):
    def __init__(self):

        self.json_path = "./archive/coco2017/annotations/instances_val2017.json"
        self.img_path = "./archive/coco2017/val2017"
        
        # load coco data
        self.coco = COCO(annotation_file=self.json_path)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(list(self.coco.imgs.keys()))
    
    def get_segmentaion(self, img, targets):
        img_w, img_h = img.size
        masks = []
        cats = []
        for target in targets:
            cats.append(target["category_id"])  # get object class id
            polygons = target["segmentation"]   # get object polygons
            rles = coco_mask.frPyObjects(polygons, img_h, img_w)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = mask.any(axis=2)
            masks.append(mask)

        cats = np.array(cats, dtype=np.int32)
        if masks:   masks = np.stack(masks, axis=0)
        else:       masks = np.zeros((0, img_h, img_w), dtype=np.uint8)

        # merge all instance masks into a single segmentation map
        # with its corresponding categories
        if(len(masks)==0):
            target = np.zeros([img_h, img_w])
        else:
            target = (masks * cats[:, None, None]).max(axis=0)
            # discard overlapping instances
            target[masks.sum(0) > 1] = 0

        target = Image.fromarray(target.astype(np.uint8))
        
        return target


    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        targets = self.coco.loadAnns(ann_ids)

        # get image file name
        path = self.coco.loadImgs(img_id)[0]['file_name']

        # read image
        img =  Image.open(os.path.join(self.img_path, path)).convert('RGB')

        image_array = np.array(img)

        # get sementation labels
        label = np.array(self.get_segmentaion(img, targets))


        image_array.resize((60, 40, 3))
        label.resize((60, 40))

        
        return image_array, label



