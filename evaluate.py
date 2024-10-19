import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.chdir('/data_hdd2/users/hassan/projects/acquarium')

import torch, argparse, tqdm, time, cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms import v2
from models.resnet_18_CNN import ResNet50_FasterRCNN
from utils.dataloader import Fish
from torch.utils.data import DataLoader

def collate_fn(batch):
    return tuple(zip(*batch))

def show(imags, coordinates):
    out_imgs = []
    for img, coords in zip(imags, coordinates):
        img = torch.permute(img, (1,2,0)).detach().cpu().numpy() 
        coords = tuple(list(coords.values())[0])
        labelled_img = img.copy()
        for c in coords:
            c = c.detach().cpu().numpy()
            cv2.rectangle(labelled_img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0,0,0), 2)
        out_imgs.append(labelled_img)
    return out_imgs

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou_value = inter_area / float(box1_area + box2_area - inter_area)
    return iou_value

def evaluate(NUM_CLASSES, BATCH_SIZE, TEST_DIR_IMG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50_FasterRCNN(NUM_CLASSES)
    model.load_state_dict(torch.load("/data_hdd1/hassan/projects/acquarium/chkpts_CNN_CUSTOM/model_35.pth", map_location="cpu")['model_state_dict'])
    model.to(device)
    model.eval()

    LABELS = {0: 'creatures', 1: 'fish', 2: 'jellyfish', 3: 'penguin', 4: 'puffin', 5: 'shark', 6: 'starfish', 7: 'stingray'}
    
    # permute_transform = transforms.Lambda(lambda x: torch.permute(x, (1, 2, 0)))
    transform = transforms.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    test_data = Fish(TEST_DIR_IMG, split='valid')
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)
    total = 0
    correct = 0
    with torch.no_grad():
        imags = []
        for _, (image, targets) in enumerate(tqdm.tqdm(test_loader)):
            test_img, boxes_pred, labels_pred  = [], [], []
            for i in image:
                imags.append(i.to(device))
            test_img.append(imags[0].detach().cpu())
            if len(test_img) == 0:
                continue
            for i in range(len(targets)):
                for k, v in targets[i].items():
                    targets[i][k] = v.to(device)
            try:
                pred = model(imags, targets)
                for j, p in enumerate(pred[0]['scores']):
                    if p > 0.5:
                        boxes_pred.append(pred[0]['boxes'][j].detach().cpu().numpy())
                        labels_pred.append(pred[0]['labels'][j].detach().cpu().numpy())
                img = torch.permute(test_img[0], (1,2,0)).numpy()
                final_img = img.copy()
                gt = targets[0]['boxes'].detach().cpu().numpy()
                labels = targets[0]['labels'].detach().cpu().numpy()
                print(pred)
                val_img = show(test_img, pred)
                plt.imshow(val_img[0])
                plt.show()

                for i in range(len(boxes_pred)):
                    total += 1
                    for j in range(len(gt)):
                        if iou(boxes_pred[i], gt[j]) > 0.5 and labels_pred[i] == labels[j]:
                            correct += 1
                imags = []

            except AssertionError:
                continue
            
            imags = []
        print(f"\nTest Accuracy of Model: {correct}\n")
    return boxes_pred, labels_pred

if __name__ == '__main__':
    NUM_CLASSES = 8
    BATCH_SIZE = 1
    TEST_DIR_IMG = r"/data_hdd1/hassan/projects/acquarium/data/acqurium/"
    boxes_pred, labels_pred = evaluate(NUM_CLASSES, BATCH_SIZE, TEST_DIR_IMG)
