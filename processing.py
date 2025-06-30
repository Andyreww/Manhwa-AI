# processing.py
# This file contains all the core logic for bubble detection, OCR, and translation.
# --- VERSION 13.0: Simplified Logic ---

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import json
import requests
from collections import Counter
from dotenv import load_dotenv

# --- Load environment variables from .env file ---
load_dotenv()

# --- CONFIGURATION ---
MODEL_PATH = 'bubble_detector.pth'
DEFAULT_FONT_PATH = "Bangers-Regular.ttf"
CONFIDENCE_THRESHOLD = 0.90
if os.path.exists('google_key.json'):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google_key.json'


# --- MODEL AND OCR SETUP ---
def get_model(num_classes=2):
    """Loads the pre-trained bubble detection model ONCE."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print("Bubble detector model loaded successfully.")
        return model, device
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model from {MODEL_PATH}. Error: {e}")
        return None, None

def ocr_google_vision(image_cv):
    try:
        from google.cloud import vision
    except ImportError:
        print("Google Cloud Vision library not found. Please run: pip3 install google-cloud-vision")
        return "", [], None

    h, w, _ = image_cv.shape
    if h == 0 or w == 0: return "", [], None

    pixels = image_cv.reshape((-1, 3))
    pixels = np.float32(pixels)

    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)

    label_counts = Counter(labels.flatten())
    background_label = max(label_counts, key=label_counts.get)
    
    combined_mask = (labels.flatten() != background_label)
    text_mask = combined_mask.reshape(image_cv.shape[:2]).astype(np.uint8) * 255
    ocr_ready_image = cv2.bitwise_not(text_mask)
    
    client = vision.ImageAnnotatorClient()
    success, encoded_image = cv2.imencode('.png', ocr_ready_image)
    if not success: return "", [], ocr_ready_image
    
    image = vision.Image(content=encoded_image.tobytes())
    response = client.text_detection(image=image, image_context={"language_hints": ["ko", "en"]})
    
    if response.text_annotations:
        full_text = response.text_annotations[0].description.replace('\n', ' ').strip()
        polygons = [np.array([[v.x, v.y] for v in ann.bounding_poly.vertices], dtype=np.int32) for ann in response.text_annotations[1:]]
        return full_text, polygons, ocr_ready_image
    
    return "", [], ocr_ready_image


def translate_with_llm(text):
    apiKey = os.getenv("GOOGLE_AI_API_KEY")
    if not apiKey:
        print("ERROR: GOOGLE_AI_API_KEY not found. Make sure it's in your .env file.")
        return "[API Key Missing]"
        
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"
    prompt = f"You are a professional Korean-to-English manhwa translator. Create a fluent, natural translation. Do not add quotation marks or periods unless the original has '...'. If the text is junk, reply with only the word IGNORE. Korean Text: \"{text}\""
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()
        result = response.json()
        if result.get('candidates'):
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
        return "IGNORE"
    except Exception as e:
        print(f"LLM Translation failed: {e}")
        return "IGNORE"


# --- DRAWING FUNCTIONS ---
def draw_text_with_outline(draw, pos, text, font, fill, outline):
    x, y = pos
    w = 2 
    draw.multiline_text((x-w, y), text, font=font, fill=outline, align="center", anchor="mm")
    draw.multiline_text((x+w, y), text, font=font, fill=outline, align="center", anchor="mm")
    draw.multiline_text((x, y-w), text, font=font, fill=outline, align="center", anchor="mm")
    draw.multiline_text((x, y+w), text, font=font, fill=outline, align="center", anchor="mm")
    draw.multiline_text(pos, text, font=font, fill=fill, align="center", anchor="mm")

def get_optimal_font_and_wrap(text, max_w, max_h, font_family=None, font_size=None):
    font_path = font_family or DEFAULT_FONT_PATH
    padding_w = max_w * 0.10
    padding_h = max_h * 0.10
    target_w = max_w - (2 * padding_w)
    target_h = max_h - (2 * padding_h)

    def wrap_text(font_to_use):
        lines = []
        words = text.split()
        if not words: return ""
        current_line = words[0]
        for word in words[1:]:
            if font_to_use.getbbox(f"{current_line} {word}")[2] <= target_w:
                current_line += f" {word}"
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        return "\n".join(lines)

    if font_size:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"Warning: Could not load font '{font_path}'. Using default.")
            font = ImageFont.load_default(size=font_size)
        wrapped_text = wrap_text(font)
        return font, wrapped_text

    else:
        current_font_size = 90
        while current_font_size > 8:
            try:
                font = ImageFont.truetype(font_path, current_font_size)
            except IOError:
                font = ImageFont.load_default()

            wrapped_text = wrap_text(font)
            draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            if text_width < target_w and text_height < target_h:
                return font, wrapped_text
            
            current_font_size -= 4
            
        try:
            font = ImageFont.truetype(font_path, current_font_size)
        except IOError:
            font = ImageFont.load_default()
        wrapped_text = wrap_text(font)
        return font, wrapped_text


# --- MAIN PROCESSING FUNCTION ---
def process_single_image(image_path, model, device, font_family=None, font_size=None):
    if model is None or device is None: 
        print("Processing error: Model not loaded.")
        return None

    try:
        original_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Could not open image {image_path}: {e}")
        return None

    img_tensor = F.to_tensor(original_pil).to(device)
    output_pil = original_pil.copy()

    with torch.no_grad():
        prediction = model([img_tensor])

    boxes = prediction[0]['boxes'][prediction[0]['scores'] > CONFIDENCE_THRESHOLD].cpu().numpy()
    
    print(f"Found {len(boxes)} bubbles in {os.path.basename(image_path)}.")

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        
        cropped_bubble_pil = original_pil.crop((x_min, y_min, x_max, y_max))
        cropped_bubble_cv = cv2.cvtColor(np.array(cropped_bubble_pil), cv2.COLOR_RGB2BGR)

        original_text, polygons, _ = ocr_google_vision(cropped_bubble_cv)
        print(f"  Bubble #{i+1} OCR: '{original_text}'")

        if not original_text.strip(): continue

        translated_text = translate_with_llm(original_text)
        print(f"  Bubble #{i+1} Translation: '{translated_text}'")
        
        if translated_text.upper() == "IGNORE" or "[API Key Missing]" in translated_text:
            continue

        mask = np.zeros(cropped_bubble_cv.shape[:2], dtype=np.uint8)
        if polygons:
            cv2.fillPoly(mask, polygons, 255)
        else:
             h, w, _ = cropped_bubble_cv.shape
             cv2.rectangle(mask, (int(w*0.05), int(h*0.05)), (int(w*0.95), int(h*0.95)), 255, -1)
        
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=3)
        inpainted_bubble_cv = cv2.inpaint(cropped_bubble_cv, mask, 3, cv2.INPAINT_TELEA)
        inpainted_bubble_pil = Image.fromarray(cv2.cvtColor(inpainted_bubble_cv, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(inpainted_bubble_pil)
        b_w, b_h = inpainted_bubble_pil.size
        
        font, wrapped_text = get_optimal_font_and_wrap(translated_text, b_w, b_h, font_family, font_size)
        
        draw_text_with_outline(draw, (b_w/2, b_h/2), wrapped_text, font, "white", "black")

        output_pil.paste(inpainted_bubble_pil, (x_min, y_min))

    return output_pil