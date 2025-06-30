# train_model.py (Pro Version w/ Checkpointing)

import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Subset
from PIL import Image
import numpy as np
import random

# --- CONFIGURATION ---
IMAGE_FOLDER = 'manhwa_panels'
ANNOTATION_FOLDER = 'annotated_data'
# The final model will be saved with this name
OUTPUT_MODEL_NAME = 'best_bubble_detector.pth' 
NUM_EPOCHS = 10
BATCH_SIZE = 4 # Safe CPU batch size
VALIDATION_SPLIT = 0.2 # Using 20% for a reliable validation score

# --- 1. THE CUSTOM DATASET (with manual augmentation) ---
class ManhwaBubbleDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_dir, augment=False):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.augment = augment
        self.imgs = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size

        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.annotation_dir, label_name)
        
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    x, y, w, h = [int(val) for val in line.strip().split()]
                    boxes.append([x, y, x + w, y + h])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        if self.augment:
            if random.random() < 0.5:
                img = F.hflip(img)
                new_boxes = []
                for box in boxes:
                    x_min, y_min, x_max, y_max = box
                    new_x_min = img_width - x_max
                    new_x_max = img_width - x_min
                    new_boxes.append([new_x_min, y_min, new_x_max, y_max])
                boxes = torch.as_tensor(new_boxes, dtype=torch.float32)

            img = F.adjust_brightness(img, brightness_factor=random.uniform(0.9, 1.1))
            img = F.adjust_contrast(img, contrast_factor=random.uniform(0.9, 1.1))

        img = F.to_tensor(img)

        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.ones((len(boxes),), dtype=torch.int64)
        
        return img, target

    def __len__(self):
        return len(self.imgs)

# --- 2. HELPER FUNCTION ---
def collate_fn(batch):
    return tuple(zip(*batch))

# --- 3. THE MODEL ---
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- 4. THE MAIN TRAINING SCRIPT ---
if __name__ == "__main__":
    print("Starting PRO model training with Checkpointing...")

    device = torch.device('cpu')
    print(f"Using device: {device}")

    num_classes = 2
    
    full_dataset = ManhwaBubbleDataset(IMAGE_FOLDER, ANNOTATION_FOLDER, augment=True)

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(VALIDATION_SPLIT * dataset_size))
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Dataset split: {len(train_dataset)} training images, {len(val_dataset)} validation images.")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=10,
                                                   gamma=0.1)

    # --- NEW: CHECKPOINTING LOGIC ---
    best_val_loss = float('inf') # Start with a very high "best score"

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            train_loss += losses.item()
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        avg_train_loss = train_loss / len(train_loader)

        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        avg_val_loss = val_loss / len(val_loader)

        lr_scheduler.step()
        
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        print(f"  Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

        # --- NEW: CHECK IF THIS IS THE BEST MODEL SO FAR ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the model because it's the new champion
            torch.save(model.state_dict(), OUTPUT_MODEL_NAME)
            print(f"  ✨ New best score! Saving model to {OUTPUT_MODEL_NAME} ✨")

    print(f"\nTraining complete! The best model (with validation loss {best_val_loss:.4f}) is saved as '{OUTPUT_MODEL_NAME}'")
