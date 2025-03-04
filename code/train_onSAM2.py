# segment image region using fine tune model SAM2
#2024.0820 V1.0.0
import numpy as np
import torch
from models.sam2.build_sam import build_sam2
from models.sam2.sam2_image_predictor import SAM2ImagePredictor

from tqdm import tqdm
from celldatav11 import load_data_cell_forsam2

# Enable cuDNN benchmarking
torch.backends.cudnn.benchmark = True

# Initialize dataset and DataLoader
batch_size = 128
train_iter = load_data_cell_forsam2(batch_size=batch_size,  test=False)
# train_iter = load_data_cell_forsam2(batch_size=batch_size,  test=True)
# Load model
sam2_checkpoint = "sam2_hiera_checkpoints/sam2_hiera_small.pt"  # path to model weights
model_cfg = "sam2_hiera_s.yaml"  # model config
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)  # load model
predictor = SAM2ImagePredictor(sam2_model)

# Set training parameters
predictor.model.sam_mask_decoder.train(True)  # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder
learn_rt = 1e-5
weight_decay = 4e-5
optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=learn_rt, weight_decay=weight_decay)
scaler = torch.cuda.amp.GradScaler()  #mixed precision training
# Define your custom loss function, for example:
def custom_loss_fn(prd_mask, gt_mask):
    return (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log(
            (1 - prd_mask) + 0.00001)).mean()


# Training loop
def train(predictor, train_iter, loss_fn, optimizer, num_epochs, device, save_path):
    print(f"Starting training... batch size:{batch_size} ,  num_epochs: {num_epochs}")

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_iou = 0.0
        num_batches = 0
        best_loss = float('inf')

        for batch in tqdm(train_iter, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            with torch.cuda.amp.autocast():  # Cast to mixed precision
                images, masks_dict_list, input_points, input_labels = batch  # Load data batch

                if len(masks_dict_list) == 0:
                    continue  # Ignore empty batches

                for i, masks_dict in enumerate(masks_dict_list):  # Iterate over each image in the batch
                    if len(masks_dict) == 0:
                        continue  # Ignore empty batches

                    predictor.set_image(images[i])  # Apply SAM image encoder to the image

                    # Prompt encoding
                    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                        input_points[i], input_labels[i], box=None, mask_logits=None, normalize_coords=True
                    )
                    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                        points=(unnorm_coords, labels), boxes=None, masks=None
                    )

                    # Mask decoder
                    batched_mode = unnorm_coords.shape[0] > 1  # Multi-object prediction
                    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                        repeat_image=batched_mode,
                        high_res_features=high_res_features
                    )
                    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

                    # Combine all masks for the current image
                    gt_mask = {key: mask.float().to(device) for key, mask in masks_dict.items()}
                    gt_mask = torch.stack(list(gt_mask.values()))

                    prd_mask = torch.sigmoid(prd_masks[:, 0])  # Convert logit map to probability map

                    # Calculate segmentation loss using the provided loss function
                    seg_loss = loss_fn(prd_mask, gt_mask)

                    # Calculate IOU Score and Score Loss
                    inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                    iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
                    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                    loss = seg_loss + score_loss * 0.05  # Mixed losses

                    # Backpropagation and optimization
                    predictor.model.zero_grad()  # Empty gradient
                    scaler.scale(loss).backward()  # Backpropagation
                    scaler.step(optimizer)
                    scaler.update()  # Mixed precision update

                    total_loss += loss.item()
                    total_iou += iou.mean().item()
                    num_batches += 1

        # Calculate average loss and IOU for the epoch
        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches

        # Save model weights at the end of each epoch
        weight_path = save_path + f"cellsam2.pth"
        if (epoch + 1) % 50 == 0:
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(predictor.model.state_dict(), weight_path)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy (IOU): {avg_iou:.4f}")
        wandb.log({"Epoch": epoch, "Loss": avg_loss, "Accuracy": avg_iou})


import os
import wandb



base_dir = ""  # Base directory for saving model weights
if os.path.isdir("/home/zhang"):
    os.makedirs(base_dir := "/home/zhang/cellseg/cellsegV0.1/sam2weights/", exist_ok=True)
elif os.path.isdir("/home/james/extdisk"):
    os.makedirs(base_dir := "/home/james/.cache/202408cellseg/cellsegV0.1/sam2weights/", exist_ok=True)
else:
    raise Exception("No such directory exists")
save_path = base_dir

num_epochs = 3000

num_classes = 4 # 4 classes: background, cell, nucleus, and nucleolus, Currently not used
# Initialize W&B
wandb.init(project="CellSam2", config={
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": learn_rt,
    "weight_decay": weight_decay,
    "num_classes": num_classes,

})

wandb.watch(predictor.model, log="all")

# Call the train function with your loss function
train(predictor, train_iter, custom_loss_fn, optimizer, num_epochs=num_epochs, device=device, save_path=save_path)

# Finish W&B run
wandb.finish()
