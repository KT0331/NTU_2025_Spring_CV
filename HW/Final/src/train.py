# train.py

import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_model import IrisMatchModel
from IrisPairDataset import IrisPairDataset
from tqdm import tqdm

def train(model, dataloader, device, optimizer, scheduler, args, ckpt_dir):
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0
        for img1, img2, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        scheduler.step()

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved to: {ckpt_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--milestones', type=int, nargs='+', default=[8, 16, 24])
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--ckpt_dir', type=str, default='./src/checkpoints')
    parser.add_argument('--patch', action='store_true', help='Enable patch mode')
    parser.add_argument('--segmentation', action='store_true', help='Enable patch mode')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocess_mode = 'origin'

    if args.patch:
        patch = True
        preprocess_mode = 'patch'
    else:
        patch = False

    if args.segmentation:
        segmentation = True
        preprocess_mode = 'segmentation'
    else:
        segmentation = False

    ckpt_dir = os.path.join(args.ckpt_dir, preprocess_mode)
    os.makedirs(ckpt_dir, exist_ok=True)

    dataset = IrisPairDataset(file_path = args.train_file, patch = patch, segmentation = segmentation)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = IrisMatchModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

    train(model, dataloader, device, optimizer, scheduler, args, ckpt_dir)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
