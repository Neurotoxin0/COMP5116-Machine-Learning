import os
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image, ImageDraw


# Change the working directory to the location of this script
Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
print(Path)
os.chdir(Path)

# set coco data path
json_path = "./archive/coco2017/annotations/instances_val2017.json"
img_path = "./archive/coco2017/val2017"

# load coco data
coco = COCO(annotation_file=json_path)

# get all image index info
ids = list(sorted(coco.imgs.keys()))
print("number of images: {}\n".format(len(ids)))

# get all coco class labels
coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])
print("coco classes: {}\n".format(coco_classes))


# plot config
num_of_pics = 5 # number of images to show in the plot
plt_index = 1

# initialize figure
plt.figure()

# find the iamges with max number of objects
num_of_objects = []
for img_id in ids:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    targets = coco.loadAnns(ann_ids)
    num_of_objects.append(len(targets))

sorted_ids = [x for _, x in sorted(zip(num_of_objects, ids), key=lambda pair: pair[0], reverse=True)]
selected_images = sorted_ids[:num_of_pics]

# iter over the selected images
for img_id in selected_images:
    # get annotation ids of the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    
    # get annotation info of the image
    targets = coco.loadAnns(ann_ids)

    # get image file name
    path = coco.loadImgs(img_id)[0]['file_name']

    # read image
    img = Image.open(os.path.join(img_path, path)).convert('RGB')
    
    # plot original image
    plt.subplot(num_of_pics,3,plt_index)
    plt_index+=1
    plt.imshow(img)

    # plot image with bounding boxes
    plt.subplot(num_of_pics,3,plt_index)
    plt_index+=1
    draw = ImageDraw.Draw(img)

    for target in targets:
        x, y, w, h = target["bbox"]
        x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, y1), coco_classes[target["category_id"]])
    
    plt.imshow(img)
    
    # plot image with segmentation map 
    plt.subplot(num_of_pics,3,plt_index)
    plt_index+=1
    img_w, img_h = img.size
    masks = []
    cats = []

    for target in targets:
        cats.append(target["category_id"])  # get object category id
        polygons = target["segmentation"]   # get object polygons
        rles = coco_mask.frPyObjects(polygons, img_h, img_w)    # convert polygons to rles
        mask = coco_mask.decode(rles)   # convert rles to mask
        
        # some objects are "iscrowd", exclude them from the final mask
        if len(mask.shape) < 3: mask = mask[..., None]
        mask = mask.any(axis=2)
        masks.append(mask)

    # convert to numpy arrays
    cats = np.array(cats, dtype=np.int32)
    if masks:   masks = np.stack(masks, axis=0)
    else:       masks = np.zeros((0, img_h, img_w), dtype=np.uint8)

    # merge all instance masks into a single segmentation map with its corresponding categories
    target = (masks * cats[:, None, None]).max(axis=0)
    # discard overlapping instances
    target[masks.sum(0) > 1] = 255
    target = Image.fromarray(target.astype(np.uint8))

    plt.imshow(target)
    
# draw final figure
plt.show()
