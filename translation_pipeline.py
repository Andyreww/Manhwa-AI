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
import textwrap 
from google.cloud import vision # Google Cloud Vision API

# --- CONFIGURATION ---
MODEL_PATH = 'bubble_detector.pth' 
TEST_IMAGE_FOLDER = 'new_manhwa_panels_to_test' 
OUTPUT_DIR = 'final_translations' # New folder for the final output
# --- NEW: Folder for our debug images ---
DEBUG_OCR_DIR = 'debug_ocr_steps' 
CONFIDENCE_THRESHOLD = 0.90
FONT_PATH = "Bangers-Regular.ttf"
# IMPORTANT: This tells the script where to find your secret key file.
# Make sure your renamed key file is in the same folder as this script.
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google_key.json'

# --- SETUP FOLDERS ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(TEST_IMAGE_FOLDER):
    print(f"Creating folder '{TEST_IMAGE_FOLDER}'. Please put your test images in there!")
    os.makedirs(TEST_IMAGE_FOLDER)
# --- NEW: Create the debug folder ---
if not os.path.exists(DEBUG_OCR_DIR):
    os.makedirs(DEBUG_OCR_DIR)

# --- MODEL AND OCR SETUP ---
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- THE FINAL BOSS v2: OCR with Surgical Color Isolation ---
def ocr_google_vision(image_cv):
    """
    This "surgical" method uses k-means clustering to isolate the main text
    color from the background and outline. It identifies the top 3 colors,
    assumes the most common is the background and the second-most common is
    the text, and creates a clean black-on-white image of only the text.
    """
    try:
        from collections import Counter
    except ImportError:
        # This dependency is usually included with modern Python/OpenCV
        print("Could not import Counter. Please ensure you are using a standard Python environment.")
        return "", [], image_cv

    # 1. Prepare the image for k-means analysis
    pixels = image_cv.reshape((-1, 3))
    pixels = np.float32(pixels)

    # 2. Find the K dominant colors. K=3 is good for background, text fill, and outline.
    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)

    # 3. **THE SURGICAL FIX**: Identify background and text by frequency
    # We count how many pixels belong to each color cluster.
    label_counts = Counter(labels.flatten())
    
    # The most frequent color is the background.
    # The second most frequent is the text fill.
    # The least frequent is the outline or noise.
    sorted_labels = sorted(label_counts, key=label_counts.get, reverse=True)
    
    background_label = sorted_labels[0]
    text_label = sorted_labels[1]

    print(f"  > Dominant colors found: {centers.tolist()}")
    print(f"  > Identified background color: {centers[background_label].tolist()}")
    print(f"  > Identified text color: {centers[text_label].tolist()}")

    # 4. Create a new image with ONLY the text color, drawn in black.
    # Create a mask where only pixels belonging to the `text_label` are white (255).
    text_mask = np.uint8(labels.reshape(image_cv.shape[:2]) == text_label) * 255
    
    # Invert the mask. Now the text is black (0) and everything else is white (255).
    # This is the perfect, high-contrast image for the OCR.
    ocr_ready_image = cv2.bitwise_not(text_mask)

    # The final processed image to be sent to the API
    debug_image = ocr_ready_image
    
    # 5. Send to Google Cloud Vision
    client = vision.ImageAnnotatorClient()
    scale_factor = 1 # k-means works well at original resolution
    success, encoded_image = cv2.imencode('.png', debug_image)
    if not success:
        return "", [], debug_image
    content = encoded_image.tobytes()

    image = vision.Image(content=content)
    response = client.text_detection(image=image, image_context={"language_hints": ["ko", "en"]})
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f"Google Vision API Error: {response.error.message}")

    if texts and texts[0].description.strip():
        print("  > SUCCESS! Surgical k-means method worked.")
        full_text = texts[0].description.replace('\n', ' ').strip()
        word_polygons = [
            np.array([[v.x / scale_factor, v.y / scale_factor] for v in ann.bounding_poly.vertices], dtype=np.int32)
            for ann in texts[1:]
        ]
        return full_text, word_polygons, debug_image
    
    print("  > Surgical k-means method failed to produce readable text.")
    return "", [], debug_image


# --- LLM TRANSLATOR FUNCTION ---
def translate_with_llm(text_to_translate):

    """

    Translates Korean manhwa text into dramatic and natural English using an LLM.

    """

    prompt = f"""

    You are a professional Korean-to-English translator who specializes in scanlating manhwa — including fantasy, action, romance, slice-of-life, and NSFW (18+).

    You specialize in fixing broken OCR, removing watermark junk, and creating fluent, emotionally natural English translations that feel authentic to the manhwa genre.

    If a character is referred to as "ML" (short for Main Lead) in the translation, rewrite it to use natural phrasing that fits spoken English, such as "him," "the guy," or the character’s name if known. Never leave "ML" in the final translation — it's an internal label, not real dialogue.



    Return ONLY ONE of the following:

    ---

    1. **If it’s dialogue:** Translate it into smooth, expressive English. Match the tone (shy, flirty, angry, scared, etc).

    - Use natural contractions: “I’m”, “Let’s”, “That’s”, etc.

    - Make it sound like something a real manhwa character would say — casual and punchy.

    - **DO NOT** end your response with a period `.` unless the Korean ends in `...`

    - **DO NOT** use quotation marks ("") — your output will be inserted directly into the bubble.

    - Prefer natural flow: “Thanks”, “What?!”, “Tch...”, “No way—” — not stiff, robotic English.



    :white_check_mark: PUNCTUATION RULES FOR DIALOGUE:

    - Use `!`, `?`, `...`, or `—` as needed to match tone.

    - **Never** end with a period `.` unless it was `...` in the original.

    - Do not add punctuation if the Korean sentence ends flat and casual — e.g., “Thanks”, “Okay”, “Right”



    2. **If it’s a sound effect (SFX), moan, gasp, laugh, or grunt:** Translate it into an expressive sound.

    Use variety! Examples:



    - “으윽” → “Ugh...!” / “Nngh!” / “Ugh—!” / “Agh...!”

    - “아앗” → “Ahh!” / “Haaah...!” / “Nnngh...!” / “Aah...!”

    - “하하하” → “Hahaha!” / “Ha!” / “Pfft!” / “LOL!”

    - “흐흐흐” → “Heh heh...” / “Hehehe” / “Snrk...” / “Heh...”

    - “큭큭” → “Tch...” / “Hmph.” / “Heh...” / “Tsk.”

    - “흐읏” → “Ngh...” / “Hnn...” / “Mmh...!” / “Haaah...!”

    - “읏” → “Hnng...!” / “Ugh...” / “Ah...”

    - “훗” → “Hmph...” / “Heh.”

    - “헉” → “Gasp!” / “Hah?!” / “Whoa!” / “What the—?!”

    - “앗” → “Ah!” / “Whoa!” / “Wait—!”



    3. **If it's unreadable watermark junk (e.g. MANGA18FX, IImMII, MAD옷균8FXa) with NO valid Korean content**, respond with:



    **IGNORE**



    :tools: OCR FIXING & TYPO RECOVERY RULES:

    - Remove watermark gibberish but save real Korean.

    Example: “MAD옷균8FXa 자, 잠깐만요 이거튼” → use: “자, 잠깐만요 이거는”

    - Translate Korean even if there's junk after it.

    Example: “아 안 되는: . IImMII” → translate just the Korean.

    - Recover OCR typos:

    - “이거튼” → likely “이거는”

    - “유강” → likely “우측”

    - “문우l” → “흐읏”

    - “@” → “아”

    - “극극극특” → “큭큭큭큭”



    :exclamation:FINAL OUTPUT RULES (READ CAREFULLY):

    - Do **NOT** wrap the response in quotation marks.

    - Do **NOT** end your line with a period `.` unless the original Korean ended with `...`

    - Do **NOT** explain anything or list alternatives.

    - Just return the final English dialogue or the word **IGNORE**.



    Korean Text to Translate: "{text_to_translate}"

    """
    try:
        # It's better practice to load the API key from an environment variable
        # rather than hardcoding it directly in the script.
        apiKey = os.environ.get("GOOGLE_AI_API_KEY", "AIzaSyDQ8cxecPNUb9iwcxQXDQuTtUgWCEZtSzs")
        if apiKey == "PASTE_YOUR_NEW_GOOGLE_AI_API_KEY_HERE" or not apiKey:
            print("  ERROR: Please paste your new Google AI API key into the script or set it as an environment variable.")
            return "[Translation Error: Missing API Key]"
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()
        result = response.json()
        if result.get('candidates'):
            return result['candidates'][0]['content']['parts'][0]['text'].strip().replace("*", "").replace("\"", "")
        return "IGNORE"
    except Exception as e:
        print(f"  LLM Translation failed: {e}")
        return "IGNORE"

# --- TEXT DRAWING AND FONT SIZING FUNCTIONS ---
def draw_text_with_outline(draw, position, text, font, fill, outline):
    x, y = position
    w = 2 # Outline width
    draw.multiline_text((x-w, y), text, font=font, fill=outline, align="center")
    draw.multiline_text((x+w, y), text, font=font, fill=outline, align="center")
    draw.multiline_text((x, y-w), text, font=font, fill=outline, align="center")
    draw.multiline_text((x, y+w), text, font=font, fill=outline, align="center")
    draw.multiline_text((x, y), text, font=font, fill=fill, align="center")

def get_optimal_font_and_wrap(text, font_path, max_width, max_height):
    try:
        font_size = 80
        # Determine if the text should be vertical based on the bubble's aspect ratio
        is_vertical = (max_height / max_width) > 1.5 if max_width > 0 else False
        
        while font_size > 8:
            # Leave some padding inside the bubble
            padding_multiplier = 0.85
            target_width = max_width * padding_multiplier
            target_height = max_height * padding_multiplier
            
            font = ImageFont.truetype(font_path, font_size)
            
            if is_vertical:
                # For vertical text, just put each character on a new line
                wrapped_text = "\n".join(list(text.replace(" ", "")))
            else:
                # Standard text wrapping logic
                words = text.split()
                lines, current_line = [], ""
                for word in words:
                    test_line = f"{current_line} {word}".strip()
                    if font.getlength(test_line) <= target_width:
                        current_line = test_line
                    else:
                        lines.append(current_line)
                        current_line = word
                lines.append(current_line)
                wrapped_text = "\n".join(lines)
                
            draw = ImageDraw.Draw(Image.new('RGB', (1,1)))
            text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")
            
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            # Check if the wrapped text fits within our target dimensions
            if text_w < target_width and text_h < target_height:
                return font, wrapped_text, is_vertical
            
            font_size -= 4 # Reduce font size and try again
            
        # If the loop finishes, return the smallest font size tried
        return font, wrapped_text, is_vertical
    except IOError:
        # Fallback to a default font if the specified one isn't found
        print(f"Warning: Font file not found at '{font_path}'. Using default font.")
        return ImageFont.load_default(), textwrap.fill(text, width=20), False

# --- MAIN PROCESSING SCRIPT ---
def process_all_images():
    print("Loading models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = get_model(num_classes=2)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{MODEL_PATH}'")
        print("Please make sure your 'bubble_detector.pth' file is in the same folder as the script.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()
    print("Bubble detector model loaded.")

    test_images = [f for f in os.listdir(TEST_IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not test_images:
        print(f"No images found in '{TEST_IMAGE_FOLDER}'. Nothing to process.")
        return
    
    for image_name in test_images:
        print(f"\n{'='*20}\n--- Processing: {image_name} ---\n{'='*20}")
        test_image_path = os.path.join(TEST_IMAGE_FOLDER, image_name)
        
        try:
            original_img_pil = Image.open(test_image_path).convert("RGB")
        except Exception as e:
            print(f"  Could not open or read image file: {e}")
            continue

        img_tensor = F.to_tensor(original_img_pil).to(device)
        
        with torch.no_grad():
            prediction = model([img_tensor])

        boxes = prediction[0]['boxes'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        output_image_pil = original_img_pil.copy()
        
        # Filter boxes based on the confidence threshold
        high_conf_boxes = boxes[scores >= CONFIDENCE_THRESHOLD]

        if len(high_conf_boxes) == 0:
            print(f"  No bubbles found with confidence >= {CONFIDENCE_THRESHOLD}")
            continue

        for i, box in enumerate(high_conf_boxes):
            x_min, y_min, x_max, y_max = map(int, box)
            # Crop the detected bubble from the original image
            cropped_bubble_pil = original_img_pil.crop((x_min, y_min, x_max, y_max))
            # Convert to OpenCV format for processing
            cropped_bubble_cv = cv2.cvtColor(np.array(cropped_bubble_pil), cv2.COLOR_RGB2BGR)
            
            # --- Use the new Google Vision OCR ---
            original_text, word_polygons, debug_image = ocr_google_vision(cropped_bubble_cv)
            
            # --- Save the debug image ---
            if debug_image is not None:
                debug_filename = f"debug_{os.path.splitext(image_name)[0]}_bubble_{i+1}.png"
                debug_path = os.path.join(DEBUG_OCR_DIR, debug_filename)
                cv2.imwrite(debug_path, debug_image)
                print(f"  Saved debug OCR image to: {debug_path}")

            print(f"\n--- Bubble #{i+1} ---")
            print(f"  Original Text (Vision API): '{original_text}'")

            if not original_text.strip():
                print("  No text found by OCR. Skipping bubble.")
                continue
            
            translated_text = translate_with_llm(original_text)

            if translated_text.strip().upper() == "IGNORE":
                print("  LLM identified as junk text. Not modifying image.")
                continue 
            
            print(f"  Final Text: '{translated_text}'")

            # Create a mask to identify the location of the original text
            mask = np.zeros(cropped_bubble_cv.shape[:2], dtype=np.uint8)
            if word_polygons:
                # Use the precise polygons from the Vision API if available
                cv2.fillPoly(mask, word_polygons, 255)
            else: # Fallback if Vision gives no polygons
                h, w, _ = cropped_bubble_cv.shape
                cv2.rectangle(mask, (int(w*0.05), int(h*0.05)), (int(w*0.95), int(h*0.95)), 255, -1)

            # Inpaint to remove the original text
            inpainted_bubble = cv2.inpaint(cropped_bubble_cv, mask, 7, cv2.INPAINT_NS)
            
            inpainted_bubble_pil = Image.fromarray(cv2.cvtColor(inpainted_bubble, cv2.COLOR_BGR2RGB))
            bubble_draw = ImageDraw.Draw(inpainted_bubble_pil)
            bubble_width, bubble_height = inpainted_bubble_pil.size
            
            font, wrapped_text, is_vertical = get_optimal_font_and_wrap(translated_text, FONT_PATH, bubble_width, bubble_height)
            
            # Center the text in the bubble
            text_bbox = bubble_draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")
            text_x = (bubble_width - (text_bbox[2] - text_bbox[0])) / 2
            text_y = (bubble_height - (text_bbox[3] - text_bbox[1])) / 2
            
            # Draw the final translated text with an outline
            draw_text_with_outline(bubble_draw, (text_x, text_y), wrapped_text, font, (255, 255, 255), (0, 0, 0))
            
            # Paste the modified bubble back onto the main image
            output_image_pil.paste(inpainted_bubble_pil, (x_min, y_min))

        output_filename = f"final_{image_name}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        output_image_pil.save(output_path)
        print(f"\nFinished {image_name}. Final version saved to {output_path}")

if __name__ == "__main__":
    process_all_images()
