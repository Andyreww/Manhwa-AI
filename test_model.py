# test_model.py

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os

# --- CONFIGURATION ---
# The path to your trained model file
MODEL_PATH = 'bubble_detector.pth' 
# The FOLDER containing all the images you want to test.
# IMPORTANT: Use images that were NOT in your training data!
TEST_IMAGE_FOLDER = 'new_manhwa_panels_to_test' 
# Where to save the final images with the boxes drawn on them
OUTPUT_DIR = 'test_results'
# Confidence threshold: Only show boxes the model is at least 80% sure about.
CONFIDENCE_THRESHOLD = 0.85

# --- SETUP FOLDERS ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Create the test image folder if it doesn't exist, so the script doesn't crash.
if not os.path.exists(TEST_IMAGE_FOLDER):
    print(f"Creating folder '{TEST_IMAGE_FOLDER}'. Please put your test images in there!")
    os.makedirs(TEST_IMAGE_FOLDER)

# --- RE-CREATE THE MODEL STRUCTURE ---
# This function needs to be identical to the one in your training script.
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None) # We load weights manually
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def process_all_images():
    """Main function to load model once and process all images in a folder."""
    # --- 1. LOAD THE TRAINED MODEL (only needs to happen once) ---
    print("Loading the trained bubble detector model...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    num_classes = 2 # background and bubble
    model = get_model(num_classes)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{MODEL_PATH}'")
        print("Did you run the training script successfully?")
        return # Exit the function if model isn't found

    model.to(device)
    model.eval() # CRITICAL! Turns off training behavior.
    print("Model loaded successfully.")

    # --- 2. LOOP THROUGH ALL TEST IMAGES ---
    test_images = [f for f in os.listdir(TEST_IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not test_images:
        print(f"No images found in '{TEST_IMAGE_FOLDER}'. Add some images and run again.")
        return

    print(f"\nFound {len(test_images)} images to test. Starting inference...")

    for image_name in test_images:
        print(f"\n--- Processing: {image_name} ---")
        test_image_path = os.path.join(TEST_IMAGE_FOLDER, image_name)

        # LOAD AND PREPARE THE TEST IMAGE
        try:
            original_img_pil = Image.open(test_image_path).convert("RGB")
        except Exception as e:
            print(f"  Could not open or process image. Error: {e}")
            continue # Skip to the next image

        img_tensor = F.to_tensor(original_img_pil).to(device)
        
        # RUN THE PREDICTION
        with torch.no_grad():
            prediction = model([img_tensor])

        # DRAW THE RESULTS
        boxes = prediction[0]['boxes'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        
        output_image = cv2.cvtColor(np.array(original_img_pil), cv2.COLOR_RGB2BGR)
        found_bubbles = 0

        for box, score in zip(boxes, scores):
            if score >= CONFIDENCE_THRESHOLD:
                found_bubbles += 1
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (34, 139, 34), 2)
                score_text = f"{score:.2f}"
                cv2.putText(output_image, score_text, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 139, 34), 2)
        
        print(f"  Found {found_bubbles} bubbles with confidence > {CONFIDENCE_THRESHOLD}")
        
        # SAVE THE RESULT
        output_filename = f"result_{image_name}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, output_image)
    
    print(f"\nDone! All test results saved in the '{OUTPUT_DIR}' folder.")

# --- MAIN INFERENCE SCRIPT ---
if __name__ == "__main__":
    process_all_images()
