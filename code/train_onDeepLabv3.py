import torchvision.models.segmentation
import torch
from tqdm import tqdm


lr = 1e-5 # the step size of the gradient descent during the training
width=height=1024 #  the dimensions of the image used for training. All images during the training processes will be resized to this size
batchSize = 8 #the number of images that will be used for each iteration of the training
weight_decay = 4e-5
scaler = torch.amp.GradScaler()  #mixed precision training
# Define your custom loss function, for example:
from celldata12deeplab3 import load_data_cell_forsam2, CELL_CLASSES

# Enable cuDNN benchmarking
torch.backends.cudnn.benchmark = True

# Initialize dataset and DataLoader
train_iter, val_iter = load_data_cell_forsam2(batch_size=batchSize,  test=False)
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True) # Load net
Net.classifier[4] = torch.nn.Conv2d(256, 4, kernel_size=(1,1), stride=(1,1)) # Change final layer to 3 classes
Net=Net.to(device)
optimizer = torch.optim.Adam(params=Net.parameters(), lr=lr) # Create adam optimizer
criterion = torch.nn.CrossEntropyLoss() # Set loss function
num_classes = len(CELL_CLASSES)
#------------Train------------------------
# Training function
def train(net, train_iter, val_iter, loss_fn, optimizer, num_epochs, device, save_path):

    print(f"开始训练... batch size: {batchSize}, num_epochs: {num_epochs}, num_classes: {num_classes}")

    best_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        net.train()
        total_loss = 0.0
        total_iou = 0.0
        num_batches = 0

        for batch in tqdm(train_iter, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            with torch.amp.autocast('cuda'):
                images, masks = batch
                imgs = images.to(device)
                gt_mask = masks.to(device)

                pred = net(imgs)['out']
                net.zero_grad()

                seg_loss = loss_fn(pred, gt_mask)

                # IoU calculation (same as before)
                pred_classes = torch.argmax(pred, dim=1)
                gt_mask = torch.clamp(gt_mask, 0, num_classes - 1)

                intersection = (pred_classes == gt_mask).float().sum(dim=(1, 2))
                pred_area = (pred_classes != 0).float().sum(dim=(1, 2))
                gt_area = (gt_mask != 0).float().sum(dim=(1, 2))
                union = pred_area + gt_area - intersection

                union = torch.maximum(union, torch.ones_like(union))
                iou = intersection / union
                iou = torch.clamp(iou, 0, 1)
                avg_iou = iou.mean()

                loss = seg_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                total_iou += avg_iou.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches

        # Validation phase
        avg_val_loss, avg_val_iou = validate(net, val_iter, loss_fn, device)

        # Save model weights if validation loss improves



        # Save model weights at the end of each epoch
        weight_path = save_path + f"celldeeplabv3_epoch{epoch + 1}.torch"
        if (epoch + 1) % 20 == 0:
            print(f'Saving model for epoch {epoch + 1}')
            torch.save(net.state_dict(), weight_path)
        best_weight_path = save_path + f"best_celldeeplabv3.torch"
        if (epoch + 1) > 20: # from 20, let's save best weights

            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"Saving model weights to {best_weight_path}")
                torch.save(net.state_dict(), best_weight_path)
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy (IOU): {avg_iou:.4f}, Val Loss: {avg_val_loss:.4f}, Val IOU: {avg_val_iou:.4f}")
        # wandb.log({"Epoch": epoch, "Loss": avg_loss, "Accuracy": avg_iou, "Val Loss": avg_val_loss, "Val IOU": avg_val_iou})



def validate(net, val_iter, loss_fn, device):
    net.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0

    with torch.no_grad():  # No need to calculate gradients during validation
        for batch in tqdm(val_iter, desc="Validating"):
            images, masks = batch
            imgs = images.to(device)
            gt_mask = masks.to(device)

            # Forward pass
            pred = net(imgs)['out']  # Predictions from model

            # Compute validation loss
            val_loss = loss_fn(pred, gt_mask)

            # Calculate IoU
            pred_classes = torch.argmax(pred, dim=1)  # Shape: [batch_size, height, width]
            gt_mask = torch.clamp(gt_mask, 0, len(CELL_CLASSES) - 1)

            # Calculate intersection and union for IoU
            intersection = (pred_classes == gt_mask).float().sum(dim=(1, 2))
            pred_area = (pred_classes != 0).float().sum(dim=(1, 2))  # Predicted area excluding background
            gt_area = (gt_mask != 0).float().sum(dim=(1, 2))  # Ground truth area excluding background
            union = pred_area + gt_area - intersection

            union = torch.maximum(union, torch.ones_like(union))  # Avoid division by zero
            iou = intersection / union
            iou = torch.clamp(iou, 0, 1)

            avg_iou = iou.mean()

            # Accumulate loss and IoU
            total_loss += val_loss.item()
            total_iou += avg_iou.item()
            num_batches += 1

    avg_val_loss = total_loss / num_batches
    avg_val_iou = total_iou / num_batches

    return avg_val_loss, avg_val_iou


import os
# import wandb



base_dir = "/home/ipsdb/"
cell_dir = os.path.join(base_dir, 'data/ips/output_patches/dataset_1')
save_path = base_dir

num_epochs = 30

# Initialize W&B
# wandb.init(project="CellSam2", config={
#     "num_epochs": num_epochs,
#     "batch_size": batchSize,
#     "learning_rate": lr,
#     "weight_decay": weight_decay,
#     "num_classes": num_classes,
#
# })

# wandb.watch(Net, log="all")

# Call the train function with your loss function
train(Net, train_iter, val_iter, criterion, optimizer=optimizer, num_epochs=num_epochs, device=device, save_path=save_path)


# Finish W&B run
# wandb.finish()
