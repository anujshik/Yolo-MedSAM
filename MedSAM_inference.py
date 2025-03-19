# -*- coding: utf-8 -*-

"""
Usage example:
python MedSAM_Inference.py -i assets/img_demo.png -o ./ --box "[[95,255,190,350], [150, 300, 220, 400]]"
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse

# Visualization functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2))

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred, size=(H, W), mode="bilinear", align_corners=False
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

#  Load model and image
parser = argparse.ArgumentParser(description="Run inference on testing set based on MedSAM")
parser.add_argument("-i", "--data_path", type=str, required=True, help="Path to input image")
parser.add_argument("-o", "--seg_path", type=str, required=True, help="Output folder for segmentation masks")
parser.add_argument("--box", type=str, required=True, help="Bounding boxes from YOLO detections")
parser.add_argument("--device", type=str, default="cpu", help="Device")
parser.add_argument("-chk", "--checkpoint", type=str, required=True, help="Path to trained model")
args = parser.parse_args()

device = args.device
medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint).to(device)
medsam_model.eval()

# Load image
img_np = io.imread(args.data_path)
if len(img_np.shape) == 2:
    img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
else:
    img_3c = img_np
H, W, _ = img_3c.shape

# Preprocessing
img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

# Parse multiple bounding boxes from YOLO detections
box_list = eval(args.box)  # Convert string input to list of bounding boxes
if not isinstance(box_list[0], list):  # If only one box, convert to list
    box_list = [box_list]

# Ensure output folder exists
os.makedirs(args.seg_path, exist_ok=True)

# Process each bounding box separately
for i, box in enumerate(box_list):
    box_np = np.array([box])  # Convert to numpy array
    box_1024 = box_np / np.array([W, H, W, H]) * 1024

    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor)

    # Run MedSAM segmentation
    medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)

    # Ensure unique output filenames per detection
    image_name = os.path.basename(args.data_path).replace(".png", "")
    segmented_output_path = os.path.join(args.seg_path, f"seg_{image_name}_{i}.png")
    visualization_output_path = os.path.join(args.seg_path, f"vis_{image_name}_{i}.png")

    io.imsave(segmented_output_path, medsam_seg.astype(np.uint8) * 255, check_contrast=False)
    print(f" Saved segmentation: {segmented_output_path}")

    # Save visualization
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_3c)
    show_box(box_np[0], ax[0])
    ax[0].set_title("Input Image and Bounding Box")

    ax[1].imshow(img_3c)
    show_mask(medsam_seg, ax[1])
    show_box(box_np[0], ax[1])
    ax[1].set_title("MedSAM Segmentation")

    plt.savefig(visualization_output_path)
    plt.close()
    print(f" Saved visualization: {visualization_output_path}")
