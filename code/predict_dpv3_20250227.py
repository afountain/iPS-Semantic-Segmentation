import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import os
from celldata12deeplab3 import CELL_CLASSES, CELL_COLORMAP, load_data_cell_forsam2

# 设置相关路径和参数
base_dir = ""
result_dir = ""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint = os.path.join(base_dir, "celldeeplabv3_epoch20.torch")
version = 'dpv3'  # 定义版本号


# 加载模型和数据
def load_model_and_data(checkpoint, device):
    # 加载模型
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, len(CELL_CLASSES), kernel_size=(1, 1), stride=(1, 1))
    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data)
    model = model.to(device)
    model.eval()  # 设置为评估模式

    # 加载数据（此处加载测试集）
    _, val_iter = load_data_cell_forsam2(batch_size=2, test=True)
    return model, val_iter


def visualize_predictions(net, data_loader, device, result_dir, version):
    net.eval()  # 设定为评估模式
    # 创建保存结果的目录：result_dir/version/
    save_dir = os.path.join(result_dir, version)
    os.makedirs(save_dir, exist_ok=True)

    image_id = 1  # 用于生成文件名的图片编号
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images, masks = batch  # 获取图像和对应的掩码
            imgs = images.to(device)
            # 获得预测结果
            pred = net(imgs)['out']
            pred_classes = torch.argmax(pred, dim=1)  # 获取预测的类别

            # 对每张图片处理
            for i in range(pred_classes.shape[0]):
                pred_class_map = pred_classes[i].cpu().numpy()  # 转换为 numpy 数组
                image = imgs[i].cpu().permute(1, 2, 0).numpy()  # (H,W,C)

                # 构造彩色分割图
                seg_map = np.zeros((pred_class_map.shape[0], pred_class_map.shape[1], 3), dtype=np.uint8)
                for class_id, color in enumerate(CELL_COLORMAP):
                    seg_map[pred_class_map == class_id] = color

                # 构造叠加图：原图与分割图叠加
                overlay_img = (0.6 * image + 0.4 * seg_map).astype(np.uint8)

                # 构造图像展示：可选，亦可直接保存
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(seg_map)
                axs[0].set_title(f"Predicted Segmentation {image_id}")
                axs[0].axis('off')
                axs[1].imshow(overlay_img)
                axs[1].set_title(f"Overlay {image_id}")
                axs[1].axis('off')
                axs[2].imshow(image.astype(np.uint8))
                axs[2].set_title(f"Original Image {image_id}")
                axs[2].axis('off')

                # 保存图片，文件名格式：result_{version}_{id}.jpg
                save_path = os.path.join(save_dir, f"result_{version}_{image_id}.jpg")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)

                print(f"Saved prediction: {save_path}")
                image_id += 1

                # 这里可以选择只处理一个 batch，或继续处理所有 batch
                # 若只处理一个 batch，则在此处加上 return


model, val_iter = load_model_and_data(checkpoint, device)
visualize_predictions(model, val_iter, device, result_dir, version)
