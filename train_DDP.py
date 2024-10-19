import os
os.chdir('/data_hdd2/users/hassan/projects/')

import torch, sys, tqdm, time, cv2, wandb
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
# from torchvision.transforms import v2
from models.resnet_18_DCNN import ResNet50_FasterRCNN
from utils.dataloader import Fish
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

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

def train(NUM_CLASSES, EPOCHS, BATCH_SIZE, LR, MOMENTUM, WEIGHT_DECAY, GAMMA, STEP_SIZE, SAVE_DIR, TRAIN_DIR_IMG, VAL_DIR_IMG, gpu_id):
    model = ResNet50_FasterRCNN(NUM_CLASSES)
    model.to(gpu_id)
    model = DDP(model, device_ids=[gpu_id])
    model.train()
    
    train_data = Fish(TRAIN_DIR_IMG, split='train')
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn, sampler=DistributedSampler(train_data))
    
    val_data = Fish(VAL_DIR_IMG, split='valid')
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn, sampler=DistributedSampler(val_data))
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    for epoch in range(EPOCHS):
        # Reset loss lists at the start of each epoch
        classifier_loss_train, regression_loss_train, objectness_loss_train, rpn_loss_train, total_loss = [], [], [], [], []

        model.train()
        for images, targets in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            imags = [img.to(gpu_id) for img in images]  # Send images to the device
            targets = [{k: v.to(gpu_id) for k, v in t.items()} for t in targets]  # Send targets to the device
            
            optimizer.zero_grad()
            loss_dict = model(imags, targets)  # Forward pass

            # Accumulate losses and convert them to scalar values using `.item()`
            classifier_loss_train.append(loss_dict['loss_classifier'].item())
            regression_loss_train.append(loss_dict['loss_box_reg'].item())
            objectness_loss_train.append(loss_dict['loss_objectness'].item())
            rpn_loss_train.append(loss_dict['loss_rpn_box_reg'].item())
            
            losses = sum(loss for loss in loss_dict.values())
            total_loss.append(losses)

            losses.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

        # Step the learning rate scheduler
        scheduler.step()

        # Log training losses
        wandb.log({
            "classifier_loss_train": sum(classifier_loss_train) / len(classifier_loss_train),
            "regression_loss_train": sum(regression_loss_train) / len(regression_loss_train),
            "objectness_loss_train": sum(objectness_loss_train) / len(objectness_loss_train),
            "rpn_loss_train": sum(rpn_loss_train) / len(rpn_loss_train),
            "total_loss": sum(total_loss) / len(total_loss)
            })
        
        # Validation phase
        model.eval()
        validation_images = []
        with torch.no_grad():
            for images, targets in tqdm.tqdm(val_loader, desc="Validation"):
                validation_images = [img.to(gpu_id) for img in images]  # Send images to the device
                targets = [{k: v.to(gpu_id) for k, v in t.items()} for t in targets]  # Send targets to the device
                
                # Inference on validation data
                predictions = model(validation_images)
                print(predictions)  # Check predictions for debugging
                
                # Optionally visualize results using `wandb`
                bb_img = show(validation_images, predictions)  # Function to display bounding boxes on image
                for i, img in enumerate(bb_img):
                    wandb.log({
                        f"validation_image_{i}": wandb.Image(np.asarray(img), caption=f"Epoch {epoch} Image {i}")
                    })
                break  # Remove this `break` to evaluate on the entire validation set
        
        # Save model checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses.item(),  # Save the scalar loss value
        }, SAVE_DIR + f"model_{epoch}.pth")

def main(rank, world_size):
    NUM_CLASSES = 8
    EPOCHS = 50
    BATCH_SIZE = 5
    LR = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    GAMMA = 0.1
    STEP_SIZE = 5
    SAVE_DIR = "/data_hdd2/users/hassan/projects/acquarium/chkpts_DCNN_CUSTOM/"
    
    IMG_DIR_TRAIN = "/data_hdd2/users/hassan/projects/acquarium/data/acqurium/"

    IMG_DIR_VALID = "/data_hdd2/users/hassan/projects/acquarium/data/acqurium/"

    setup(rank, world_size)
    wandb.init(project="aquarium", entity="ciir", group="DCNN")
    train(NUM_CLASSES, EPOCHS, BATCH_SIZE, LR, MOMENTUM, WEIGHT_DECAY, GAMMA, STEP_SIZE, SAVE_DIR, IMG_DIR_TRAIN, IMG_DIR_VALID, rank)
    destroy_process_group()

if __name__ == '__main__':
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on: ", d)

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
