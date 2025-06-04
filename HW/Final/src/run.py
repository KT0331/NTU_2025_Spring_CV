import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path

import argparse
import numpy as np
import cv2
import torch
from torchvision import transforms

from segmentation import segment_and_unwrap
from custom_model import IrisMatchModel


def preprocess_image(img, segmentation):

    if segmentation is True:
        img_seg = segment_and_unwrap(img)
    else:
        img_seg = img
    img_rgb = cv2.cvtColor(img_seg, cv2.COLOR_GRAY2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img_rgb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ganzin Iris Recognition Challenge')
    parser.add_argument('--input', type=str, metavar='PATH', required=True,
                        help='Input file to specify a list of sampled pairs')
    parser.add_argument('--output', type=str, metavar='PATH', required=True,
                        help='A list of sampled pairs and the testing results')
    parser.add_argument('--checkpoint_file_path', type=str, metavar='PATH', required=True,
                        default="./src/custom/ckpt_weights_Lamp",
                        help='Path to model checkpoint (voter model)')
    parser.add_argument('--patch', action='store_true', help='Enable patch mode')
    parser.add_argument('--segmentation', action='store_true', help='Enable patch mode')
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_file_path) / "origin_final_model.pth"

    if args.patch:
        patch = True
        ckpt = Path(args.checkpoint_file_path) / "patch_final_model.pth"
    else:
        patch = False

    if args.segmentation:
        segmentation = True
        ckpt = Path(args.checkpoint_file_path) / "segmentation_final_model.pth"
    else:
        segmentation = False

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IrisMatchModel().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    with open(args.input, 'r') as in_file, open(args.output, 'w') as out_file:
        for line in in_file:
            lineparts = line.split(',')
            img1_path = Path(lineparts[0].strip())
            img2_path = Path(lineparts[1].strip())

            if patch is True:
                img1_path = str(Path("dataset/patch") / img1_path.relative_to("dataset"))
                img2_path = str(Path("dataset/patch") / img2_path.relative_to("dataset"))
            else:
                img1_path = str(Path("dataset/origin") / img1_path.relative_to("dataset"))
                img2_path = str(Path("dataset/origin") / img2_path.relative_to("dataset"))

            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

            # TODO: Replace with your algorithm
            # score = np.random.rand()

            # custom model
            img1_tensor = preprocess_image(img1, segmentation).unsqueeze(0).to(device)
            img2_tensor = preprocess_image(img2, segmentation).unsqueeze(0).to(device)

            with torch.no_grad():
                score = model(img1_tensor, img2_tensor).item()


            output_line = f"{img1_path}, {img2_path}, {score}"
            print(output_line)
            out_file.write(output_line.rstrip('\n') + '\n')



