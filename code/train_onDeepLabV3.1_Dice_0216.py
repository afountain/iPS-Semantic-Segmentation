import torchvision.models.segmentation
import torch
import os
import re
import pandas as pd
from tqdm import tqdm
from celldata12deeplab3 import load_data_cell_forsam2, CELL_CLASSES

# Training parameters
lr = 1e-5
width = height = 1024
batchSize = 8
weight_decay = 4e-5
scaler = torch.amp.GradScaler()

torch.backends.cudnn.benchmark = True

# Initialize dataset and DataLoader
train_iter, val_iter = load_data_cell_forsam2(batch_size=batchSize, test=False)
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

# Load model without pretrained weights (and avoid downloading any weight)
Net = torchvision.models.segmentation.deeplabv3_resnet50(
    weights=None,           # 不使用任何预训练权重
    weights_backbone=None,  # 不使用预训练的backbone
    progress=False          # 禁用下载进度
)
# 如果你的 TorchVision 版本不支持上述参数，可尝试：
# Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=False)

# 修改分类头为 4 类
Net.classifier[4] = torch.nn.Conv2d(256, 4, kernel_size=(1,1), stride=(1,1))
Net = Net.to(device)

optimizer = torch.optim.Adam(params=Net.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
num_classes = len(CELL_CLASSES)

def save_log(log_path, logs):
    """
    将当前训练所有 epoch 的信息（logs）写入单个 CSV 文件：train<id>.csv
    logs 为 list，每个元素形如:
      (epochid, loss, trainIOU, trainDice, valid loss, validIOU, validDice)
    """
    os.makedirs(log_path, exist_ok=True)
    # 查找已有的 train<id>.csv 文件，以免覆盖历史文件
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

    # === NEW === 在 DataFrame 中增加 trainDice、validDice 列
    df = pd.DataFrame(logs, columns=[
        "epochid", "loss", "trainIOU", "trainDice", "valid loss", "validIOU", "validDice"
    ])
    df.to_csv(file_path, index=False)

def validate(net, val_iter, loss_fn, device):
    net.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0  # === NEW === 验证集 Dice 累积
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_iter, desc="Validating"):
            images, masks = batch
            imgs = images.to(device)
            gt_mask = masks.to(device)
            pred = net(imgs)['out']

            val_loss = loss_fn(pred, gt_mask)

            pred_classes = torch.argmax(pred, dim=1)
            gt_mask = torch.clamp(gt_mask, 0, len(CELL_CLASSES) - 1)

            # 计算 IOU
            intersection = (pred_classes == gt_mask).float().sum(dim=(1, 2))
            pred_area = (pred_classes != 0).float().sum(dim=(1, 2))
            gt_area = (gt_mask != 0).float().sum(dim=(1, 2))
            union = pred_area + gt_area - intersection
            union = torch.maximum(union, torch.ones_like(union))
            iou = intersection / union
            iou = torch.clamp(iou, 0, 1)
            avg_iou = iou.mean()

            # === NEW === 计算 Dice Score
            # Dice = 2 * intersection / (pred_area + gt_area + 1e-6)
            dice = 2.0 * intersection / (pred_area + gt_area + 1e-6)
            avg_dice = dice.mean()

            total_loss += val_loss.item()
            total_iou += avg_iou.item()
            total_dice += avg_dice.item()
            num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0

    avg_val_loss = total_loss / num_batches
    avg_val_iou = total_iou / num_batches
    avg_val_dice = total_dice / num_batches  # === NEW === 验证集平均 Dice
    return avg_val_loss, avg_val_iou, avg_val_dice

def train(net, train_iter, val_iter, loss_fn, optimizer, num_epochs, device, save_path, log_path):
    print(f"开始训练... batch size: {batchSize}, num_epochs: {num_epochs}, num_classes: {num_classes}")
    best_loss = float('inf')

    # 用于收集所有 epoch 的训练结果
    # === NEW === 增加 trainDice、validDice
    all_logs = []  # 每个元素形如 (epochid, train_loss, trainIOU, trainDice, val_loss, valIOU, valDice)

    for epoch in range(num_epochs):
        net.train()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0  # === NEW === 训练过程中的 Dice 累积
        num_batches = 0

        for batch in tqdm(train_iter, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            with torch.amp.autocast('cuda'):
                images, masks = batch
                imgs = images.to(device)
                gt_mask = masks.to(device)

                pred = net(imgs)['out']
                net.zero_grad()
                seg_loss = loss_fn(pred, gt_mask)

                pred_classes = torch.argmax(pred, dim=1)
                gt_mask = torch.clamp(gt_mask, 0, num_classes - 1)

                # 计算 IOU
                intersection = (pred_classes == gt_mask).float().sum(dim=(1, 2))
                pred_area = (pred_classes != 0).float().sum(dim=(1, 2))
                gt_area = (gt_mask != 0).float().sum(dim=(1, 2))
                union = pred_area + gt_area - intersection
                union = torch.maximum(union, torch.ones_like(union))
                iou = intersection / union
                iou = torch.clamp(iou, 0, 1)
                avg_iou = iou.mean()

                # === NEW === 计算 Dice
                # Dice = 2 * intersection / (pred_area + gt_area + 1e-6)
                dice = 2.0 * intersection / (pred_area + gt_area + 1e-6)
                avg_dice = dice.mean()

                loss = seg_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                total_iou += avg_iou.item()
                total_dice += avg_dice.item()
                num_batches += 1

        if num_batches == 0:
            avg_loss, avg_iou, avg_dice = 0.0, 0.0, 0.0
        else:
            avg_loss = total_loss / num_batches
            avg_iou = total_iou / num_batches
            avg_dice = total_dice / num_batches

        # 验证集
        avg_val_loss, avg_val_iou, avg_val_dice = validate(net, val_iter, loss_fn, device)

        # === NEW === 收集该 epoch 的信息 (trainDice, validDice)
        all_logs.append((
            epoch + 1,
            avg_loss,
            avg_iou,
            avg_dice,
            avg_val_loss,
            avg_val_iou,
            avg_val_dice
        ))

        # 定期保存或根据需求保存权重
        weight_path = os.path.join(save_path, f"celldeeplabv3_epoch{epoch + 1}.torch")
        if (epoch + 1) % 20 == 0:
            print(f'Saving model for epoch {epoch + 1}')
            torch.save(net.state_dict(), weight_path)

        best_weight_path = os.path.join(save_path, "best_celldeeplabv3.torch")
        if (epoch + 1) > 20 and avg_loss < best_loss:
            best_loss = avg_loss
            print(f"Saving model weights to {best_weight_path}")
            torch.save(net.state_dict(), best_weight_path)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f}, Train IOU: {avg_iou:.4f}, Train Dice: {avg_dice:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val IOU: {avg_val_iou:.4f}, Val Dice: {avg_val_dice:.4f}"
        )

    # 训练结束后，将所有 epoch 的日志写入单独一个 CSV 文件
    save_log(log_path, all_logs)

# Base directory setup
base_dir = "/home/ipsdb/"
cell_dir = os.path.join(base_dir, 'data/ips/output_patches/dataset_1')
os.makedirs(base_dir, exist_ok=True)
log_dir = os.path.join(base_dir, "logs")
num_epochs = 50

train(Net, train_iter, val_iter, criterion, optimizer, num_epochs, device, base_dir, log_dir)
