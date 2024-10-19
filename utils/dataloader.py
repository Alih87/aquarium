import torch, cv2, os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from pycocotools.coco import COCO
from torchvision import datasets
import matplotlib.pyplot as plt
import copy

def show(imags, coordinates):
    out_imgs = []
    for img, coords in zip(imags, coordinates):
        img = img.detach().cpu().numpy()[0]
        coords = tuple(list(coords.values())[0])
        labelled_img = img.copy()
        for c in coords:
            c = c.detach().cpu().numpy()
            cv2.rectangle(labelled_img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0,0,0), 2)
        out_imgs.append(labelled_img)
    return out_imgs

def collate_fn(batch):
    return tuple(zip(*batch))

class Fish(datasets.VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, transforms=None):
        # the 3 transform parameters are reuqired for datasets.VisionDataset
        super().__init__(root, transforms, transform, target_transform)
        self.split = split #train, valid, test
        self.coco = COCO(os.path.join(root, split, "_annotations.coco.json")) # annotatiosn stored here
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]
    
    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, self.split, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        target = copy.deepcopy(self._load_target(id))
        
        boxes = [t['bbox'] + [t['category_id']] for t in target] # required annotation format for albumentations
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)
            image = transformed['image']
            boxes = transformed['bboxes']
        
        new_boxes = [] # convert from xywh to xyxy
        for box in boxes:
            xmin = box[0]
            xmax = xmin + box[2]
            ymin = box[1]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.tensor(new_boxes, dtype=torch.float32)
        
        targ = {} # here is our transformed target
        targ['boxes'] = boxes
        targ['labels'] = torch.tensor([t['category_id'] for t in target], dtype=torch.int64)
        targ['image_id'] = torch.tensor([t['image_id'] for t in target])
        targ['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # we have a different area
        targ['iscrowd'] = torch.tensor([t['iscrowd'] for t in target], dtype=torch.int64)
        return torch.permute(torch.tensor(image/255.0).to(torch.float), (2,0,1)), targ # scale images
    
    def __len__(self):
        return len(self.ids)
    
if __name__ == "__main__":
    fish = Fish("/data_hdd1/hassan/projects/acquarium/data/acqurium", split='train')
    train_loader = DataLoader(fish, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    for i, (images, targets) in enumerate(train_loader):
        imags = []
        for i in images:
                imags.append(i)
        for i in range(len(targets)):
            for k, v in targets[i].items():
                targets[i][k] = v
    bb_img = show(imags, targets)
    plt.imshow(bb_img[0])