import torch
import os
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import re

# === NEW === 导入保存日志所需模块
from models.sam2.build_sam import build_sam2
from models.sam2.sam2_image_predictor import SAM2ImagePredictor

from celldatav21medsam2 import load_data_cell_forsam2_2
from utils import *
#disable flash to remove warning 20250217
torch.backends.cuda.enable_flash_sdp(False)

# Enable cuDNN benchmarking
torch.backends.cudnn.benchmark = True
# enable cuDNN SDPA
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"


# === NEW === 日志保存函数
def save_log(log_path, logs):
    """
    将当前训练所有 epoch 的信息写入单个 CSV 文件：train<id>.csv
    logs 为 list，每个元素形如：
      (epoch, train_loss, train_iou, train_dice, val_loss, val_iou, val_dice)
    """
    os.makedirs(log_path, exist_ok=True)
    existing_files = [f for f in os.listdir(log_path) if re.match(r'^train\d+\.csv$', f)]
    if existing_files:
        existing_ids = []
        for f in existing_files:
            match = re.match(r'^train(\d+)\.csv$', f)
            if match:
                existing_ids.append(int(match.group(1)))
        new_id = max(existing_ids) + 1
    else:
        new_id = 1

    file_name = f"train{new_id}.csv"
    file_path = os.path.join(log_path, file_name)

    df = pd.DataFrame(logs, columns=[
        "epoch", "train_loss", "train_iou", "train_dice",
        "val_loss", "val_iou", "val_dice"
    ])
    df.to_csv(file_path, index=False)
    print(f"Training log saved to {file_path}")


# Initialize dataset and DataLoader
batch_size = 2
train_iter, val_iter = load_data_cell_forsam2_2(batch_size=batch_size, test=False)

# Load model
sam2_checkpoint = "sam2_hiera_checkpoints/sam2_hiera_small.pt"  # path to model weights
model_cfg = "sam2_hiera_s.yaml"  # model config
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)  # load model
predictor = SAM2ImagePredictor(sam2_model)

# Set training parameters
predictor.model.sam_mask_decoder.train(True)  # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True)  # enable training of prompt encoder
learn_rt = 1e-5
weight_decay = 4e-5

image_size = 1024
out_size = 1024
memory_bank_size = 16
pos_weight = torch.ones([1]).to(device, non_blocking=True) * 2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
mask_type = torch.float32

optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=learn_rt, weight_decay=weight_decay)
scaler = torch.cuda.amp.GradScaler()  # mixed precision training


def custom_loss_fn(prd_mask, gt_mask):
    return (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)).mean()


num_classes = 4  # 4 classes: background, cell, nucleus, nucleolus

# 设置权重保存及样本可视化目录
server_base_dir = "/home/zhang/"
local_base_dir = "/home/james/"
if os.path.isdir(server_base_dir):
    os.makedirs(weight_dir := server_base_dir + "cellseg/cellsegV0.1/sam2weights/", exist_ok=True)
    os.makedirs(sample_path := server_base_dir + "cellseg/cellsegV0.1/samples/", exist_ok=True)
elif os.path.isdir(local_base_dir):
    os.makedirs(weight_dir := local_base_dir + ".cache/202408cellseg/cellsegV0.1/sam2weights/", exist_ok=True)
    os.makedirs(sample_path := local_base_dir + ".cache/202408cellseg/cellsegV0.1/sam2weights/", exist_ok=True)
else:
    raise Exception("No such directory exists")

save_path = weight_dir
sample_path = sample_path
print(f'current save_path: {save_path}, \n sample_path: {sample_path}')

num_epochs = 300


def train(sam2_model, train_iter, val_iter, loss_fn, optimizer, num_epochs, device, save_path, num_classes=num_classes):
    print(f"Starting training... batch size:{batch_size}, num_epochs: {num_epochs}")

    best_loss = float('inf')
    memory_bank_list = []
    feat_sizes = [(256, 256), (128, 128), (64, 64)]

    all_logs = []  # (epoch, train_loss, train_iou, train_dice, val_loss, val_iou, val_dice)

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        num_batches = 0

        sam2_model.sam_mask_decoder.train(True)
        sam2_model.sam_prompt_encoder.train(True)

        with tqdm(total=len(train_iter), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
            for batch in train_iter:
                optimizer.zero_grad()
                batch_loss = 0
                batch_iou_sum = 0.0
                batch_dice_sum = 0.0
                class_count = 0

                with torch.cuda.amp.autocast():
                    images, masks_dicts, input_points, input_labels = batch
                    images = images.to(device)

                    backbone_out = sam2_model.forward_image(images)
                    _, vision_feats, vision_pos_embeds, _ = sam2_model._prepare_backbone_features(backbone_out)
                    B = vision_feats[-1].size(1)

                    for cid in range(num_classes):
                        if cid in masks_dicts:
                            gt_mask = masks_dicts[cid].to(device)
                            # 确保 gt_mask 形状为 [B, image_size, image_size]
                            if gt_mask.numel() != B * image_size * image_size:
                                gt_mask = gt_mask.view(B, image_size, image_size)

                            points = input_points[cid].to(device)
                            input_labels_ = input_labels.reshape(B, -1).to(device)

                            to_cat_memory = []
                            to_cat_memory_pos = []
                            if len(memory_bank_list) == 0:
                                vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(
                                    torch.zeros(1, B, sam2_model.hidden_dim)
                                ).to(device=device)
                                vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(
                                    torch.zeros(1, B, sam2_model.hidden_dim)
                                ).to(device=device)
                            else:
                                for element in memory_bank_list:
                                    to_cat_memory.append(
                                        element[0].to(device, non_blocking=True).flatten(2).permute(2, 0, 1)
                                    )
                                    to_cat_memory_pos.append(
                                        element[1].to(device, non_blocking=True).flatten(2).permute(2, 0, 1)
                                    )
                                memory = torch.cat(to_cat_memory, dim=0)
                                memory_pos = torch.cat(to_cat_memory_pos, dim=0)
                                memory = memory.repeat(1, B, 1)
                                memory_pos = memory_pos.repeat(1, B, 1)

                                vision_feats[-1] = sam2_model.memory_attention(
                                    curr=[vision_feats[-1]],
                                    curr_pos=[vision_pos_embeds[-1]],
                                    memory=memory,
                                    memory_pos=memory_pos,
                                    num_obj_ptr_tokens=0
                                )

                            feats = [
                                        feat.permute(1, 2, 0).view(B, -1, *fsz)
                                        for feat, fsz in zip(vision_feats[::-1], feat_sizes[::-1])
                                    ][::-1]
                            image_embed = feats[-1]
                            high_res_feats = feats[:-1]

                            validbs = points.size(0)
                            coords_torch = points.clone().detach().to(device=device, dtype=torch.float)
                            coords_torch = coords_torch.reshape(validbs, -1, 2)
                            labels_torch = input_labels_[:, cid:cid + 1]

                            if (num_batches % 5) == 0:
                                prompt_points = (coords_torch, labels_torch)
                                flag = True
                            else:
                                prompt_points = None
                                flag = False

                            with torch.no_grad():
                                se, de = sam2_model.sam_prompt_encoder(
                                    points=prompt_points,
                                    boxes=None,
                                    masks=None,
                                    batch_size=B,
                                )

                            low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = \
                                sam2_model.sam_mask_decoder(
                                    image_embeddings=image_embed,
                                    image_pe=sam2_model.sam_prompt_encoder.get_dense_pe(),
                                    sparse_prompt_embeddings=se,
                                    dense_prompt_embeddings=de,
                                    multimask_output=False,
                                    repeat_image=False,
                                    high_res_features=high_res_feats
                                )

                            pred = F.interpolate(low_res_multimasks, size=(out_size, out_size))
                            high_res_multimasks = F.interpolate(
                                low_res_multimasks,
                                size=(image_size, image_size),
                                mode="bilinear",
                                align_corners=False
                            )

                            maskmem_features, maskmem_pos_enc = sam2_model._encode_new_memory(
                                current_vision_feats=vision_feats,
                                feat_sizes=feat_sizes,
                                pred_masks_high_res=high_res_multimasks,
                                is_mask_from_pts=flag
                            )
                            maskmem_features = maskmem_features.to(torch.bfloat16).to(device=device, non_blocking=True)
                            maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16).to(device=device, non_blocking=True)

                            if len(memory_bank_list) < memory_bank_size:
                                for mbatch in range(maskmem_features.size(0)):
                                    memory_bank_list.append([
                                        (maskmem_features[mbatch].unsqueeze(0)).detach(),
                                        (maskmem_pos_enc[mbatch].unsqueeze(0)).detach(),
                                        iou_predictions[mbatch, 0]
                                    ])
                            else:
                                for mebatch in range(maskmem_features.size(0)):
                                    memory_bank_maskmem_features_flatten = [
                                        element[0].reshape(-1) for element in memory_bank_list
                                    ]
                                    memory_bank_maskmem_features_flatten = torch.stack(
                                        memory_bank_maskmem_features_flatten)
                                    memory_bank_maskmem_features_norm = F.normalize(
                                        memory_bank_maskmem_features_flatten, p=2, dim=1)
                                    current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm,
                                                                         memory_bank_maskmem_features_norm.t())
                                    current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                                    diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                                    current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')
                                    single_key_norm = F.normalize(maskmem_features[mebatch].reshape(-1), p=2,
                                                                  dim=0).unsqueeze(1)
                                    similarity_scores = torch.mm(memory_bank_maskmem_features_norm,
                                                                 single_key_norm).squeeze()
                                    min_similarity_index = torch.argmin(similarity_scores)
                                    max_similarity_index = torch.argmax(
                                        current_similarity_matrix_no_diag[min_similarity_index])
                                    if similarity_scores[min_similarity_index] < \
                                            current_similarity_matrix_no_diag[min_similarity_index][
                                                max_similarity_index]:
                                        if iou_predictions[mebatch, 0] > memory_bank_list[max_similarity_index][
                                            2] - 0.1:
                                            memory_bank_list.pop(max_similarity_index)
                                            memory_bank_list.append([
                                                (maskmem_features[mebatch].unsqueeze(0)).detach(),
                                                (maskmem_pos_enc[mebatch].unsqueeze(0)).detach(),
                                                iou_predictions[mebatch, 0]
                                            ])

                            # 计算 loss（注意 loss 计算用 logits，所以不应用 sigmoid）
                            pred = pred.float()
                            gt_mask = gt_mask.float()
                            seg_loss = loss_fn(pred, gt_mask)  # loss函数要求目标形状与预测一致
                            batch_loss += seg_loss

                            # === NEW === 计算 IOU & Dice
                            epsilon = 1e-6
                            # 先将 logits 转为概率，然后二值化
                            pred_prob = torch.sigmoid(pred)
                            pred_bin = (pred_prob > 0.5).float().squeeze(1)
                            # 确保 gt_mask 形状为 [B, image_size, image_size]
                            gt_mask = gt_mask.view(B, image_size, image_size)
                            inter = (gt_mask * pred_bin).sum(dim=(1, 2))
                            union = gt_mask.sum(dim=(1, 2)) + pred_bin.sum(dim=(1, 2)) - inter
                            iou = inter / (union + 1e-6)
                            dice = 2.0 * inter / (gt_mask.sum(dim=(1, 2)) + pred_bin.sum(dim=(1, 2)) + 1e-6)
                            batch_iou_sum += iou.mean().item()
                            batch_dice_sum += dice.mean().item()
                            class_count += 1
                    if class_count > 0:
                        total_iou += batch_iou_sum / class_count
                        total_dice += batch_dice_sum / class_count

                    batch_loss.backward()
                    optimizer.step()

                total_loss += batch_loss.item()
                num_batches += 1
                pbar.set_postfix({'loss (batch)': batch_loss.item()})
                pbar.update()

        avg_loss = total_loss / num_batches if num_batches else 0.0
        avg_iou = total_iou / num_batches if num_batches else 0.0
        avg_dice = total_dice / num_batches if num_batches else 0.0

        val_loss, (val_iou, val_dice) = validation_sam(val_iter, epoch, sam2_model)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}, IOU: {avg_iou:.4f}, Dice: {avg_dice:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val IOU: {val_iou:.4f}, Val Dice: {val_dice:.4f}"
        )

        all_logs.append((epoch + 1, avg_loss, avg_iou, avg_dice, val_loss, val_iou, val_dice))

        if val_loss < best_loss and val_loss < 1:
            best_loss = val_loss
            weight_path = os.path.join(save_path, f"cell_medsam2_valoss{val_loss}.torch")
            print(f"Saving model weights to {weight_path}")
            torch.save(sam2_model.state_dict(), weight_path)

    log_dir = os.path.join(save_path, "logsMedSAM2")
    save_log(log_dir, all_logs)


import math


def validation_sam(val_loader, epoch, net: nn.Module, clean_dir=True):
    from torch.cuda.amp import autocast
    net.eval()
    n_val = len(val_loader)

    total_iou = 0.0
    total_dice = 0.0
    total_class_count = 0  # 统计处理的类别数
    device = next(net.parameters()).device
    # 与训练保持一致的特征尺寸
    feat_sizes = [(256, 256), (128, 128), (64, 64)]
    num_classes = 4  # 遍历所有类别

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, batch in enumerate(val_loader):
            with torch.no_grad(), autocast():
                total_loss = 0.0
                images, masks_dicts, input_points, input_labels = batch
                images = images.to(device)
                # 前向编码
                backbone_out = net.forward_image(images)
                _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
                B = vision_feats[-1].size(1)
                # 重构特征，与训练函数一致
                N = vision_feats[-1].shape[0]
                H_emb = W_emb = int(math.sqrt(N))
                feats = [feat.permute(1, 2, 0).view(B, -1, *fsz)
                         for feat, fsz in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
                image_embeddings = feats[-1]  # 低分辨率特征
                high_res_feats = feats[:-1]  # 高分辨率特征列表（通常为2个元素）

                # 获取 dense positional embedding，不重复
                dense_pe = net.sam_prompt_encoder.get_dense_pe()

                # 遍历所有类别
                for cid in range(num_classes):
                    if cid not in masks_dicts:
                        continue
                    # 处理目标掩码：转换为 float 并归一化（假设如果最大值大于1则需要除以255）
                    gt_mask = masks_dicts[cid].to(device).float()
                    if gt_mask.max() > 1:
                        gt_mask = gt_mask / 255.0
                    if gt_mask.numel() != B * image_size * image_size:
                        gt_mask = gt_mask.view(B, image_size, image_size)

                    # 获取提示信息
                    points = input_points[cid].to(device)
                    input_labels_ = input_labels.reshape(B, -1).to(device)
                    coords_torch = points.clone().detach().to(device, dtype=torch.float)
                    coords_torch = coords_torch.reshape(points.size(0), -1, 2)
                    labels_torch = input_labels_[:, cid:cid + 1]

                    # 通过提示编码器得到提示嵌入
                    se, de = net.sam_prompt_encoder(
                        points=(coords_torch, labels_torch),
                        boxes=None,
                        masks=None,
                        batch_size=B,
                    )

                    # 调用 mask_decoder 得到预测
                    low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = \
                        net.sam_mask_decoder(
                            image_embeddings=image_embeddings,
                            image_pe=dense_pe,
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de,
                            multimask_output=False,
                            repeat_image=False,
                            high_res_features=high_res_feats
                        )
                    # 将低分辨率预测上采样到目标尺寸
                    pred = F.interpolate(low_res_multimasks, size=(out_size, out_size))

                    # 计算 loss，保持与训练一致：训练时使用 loss_fn(pred, gt_mask)
                    lossfunc = criterion_G
                    cur_loss = lossfunc(pred, gt_mask)  # 注意：这里不使用 unsqueeze，因为训练中也没有使用
                    total_loss += cur_loss.item()

                    # 计算指标：先用 sigmoid 转换预测，然后二值化
                    pred_prob = torch.sigmoid(pred)
                    pred_bin = (pred_prob > 0.5).float().squeeze(1)  # shape: [B, H, W]
                    gt_mask_reshaped = gt_mask.view(B, image_size, image_size)
                    inter = (gt_mask_reshaped * pred_bin).sum(dim=(1, 2))
                    union = gt_mask_reshaped.sum(dim=(1, 2)) + pred_bin.sum(dim=(1, 2)) - inter
                    iou = inter / (union + 1e-6)
                    dice = 2.0 * inter / (gt_mask_reshaped.sum(dim=(1, 2)) + pred_bin.sum(dim=(1, 2)) + 1e-6)
                    total_iou += iou.mean().item()
                    total_dice += dice.mean().item()
                    total_class_count += 1
                pbar.update()

    avg_val_loss = total_loss / n_val if n_val > 0 else 0.0
    avg_val_iou = total_iou / total_class_count if total_class_count > 0 else 0.0
    avg_val_dice = total_dice / total_class_count if total_class_count > 0 else 0.0
    return avg_val_loss, (avg_val_iou, avg_val_dice)


# 开始训练
train(
    sam2_model,
    train_iter,
    val_iter,
    criterion_G,
    optimizer,
    num_epochs=num_epochs,
    device=device,
    save_path=save_path,
    num_classes=num_classes
)
