# annotation_tool.py

import cv2
import os
import json

# --- CONFIGURATION ---
# The folder with all your raw manhwa panel images
IMAGE_SOURCE_FOLDER = 'manhwa_panels' 
# The folder where we'll save the annotated images and label files
OUTPUT_ANNOTATIONS_FOLDER = 'annotated_data' 
# The name of the file that will keep track of our progress
PROGRESS_FILE = 'annotation_progress.json'

# --- GLOBAL VARIABLES ---
drawing = False  # True if mouse is pressed
ix, iy = -1, -1 # Starting x, y coordinates
current_boxes = [] # List to store boxes for the current image
current_image = None
display_image = None

# --- SETUP FOLDERS ---
if not os.path.exists(IMAGE_SOURCE_FOLDER):
    print(f"Creating folder '{IMAGE_SOURCE_FOLDER}'. Go put your manhwa JPEGs in there!")
    os.makedirs(IMAGE_SOURCE_FOLDER)

if not os.path.exists(OUTPUT_ANNOTATIONS_FOLDER):
    os.makedirs(OUTPUT_ANNOTATIONS_FOLDER)

def load_progress():
    """Loads the list of already annotated files."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_progress(annotated_files):
    """Saves the list of annotated files."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(annotated_files, f)

def draw_boxes_on_image(image, boxes):
    """Helper function to draw all boxes on the image."""
    local_display_image = image.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(local_display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return local_display_image

# --- MOUSE CALLBACK FUNCTION ---
def draw_rectangle(event, x, y, flags, param):
    """Callback function for mouse events to draw rectangles."""
    global ix, iy, drawing, current_image, display_image, current_boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Create a clean image to draw the new rectangle on
            img_copy = draw_boxes_on_image(current_image, current_boxes)
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 0, 255), 2)
            display_image = img_copy

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Finalize the rectangle
        w = abs(ix - x)
        h = abs(iy - y)
        start_x = min(ix, x)
        start_y = min(iy, y)
        
        # Only add the box if it has a noticeable size
        if w > 5 and h > 5:
            current_boxes.append((start_x, start_y, w, h))
            print(f"Added box: x={start_x}, y={start_y}, w={w}, h={h}")
        
        # Update the display with the permanent green box
        display_image = draw_boxes_on_image(current_image, current_boxes)


# --- MAIN ANNOTATION LOOP ---
def start_annotation():
    """Main loop to go through images and annotate them."""
    global current_image, display_image, current_boxes
    
    annotated_files = load_progress()
    all_images = [f for f in os.listdir(IMAGE_SOURCE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images_to_process = [f for f in all_images if f not in annotated_files]
    
    if not images_to_process:
        print("All images in the source folder have been annotated! You're a legend.")
        return

    cv2.namedWindow('Annotation Tool')
    cv2.setMouseCallback('Annotation Tool', draw_rectangle)

    for image_name in images_to_process:
        image_path = os.path.join(IMAGE_SOURCE_FOLDER, image_name)
        current_image = cv2.imread(image_path)
        
        if current_image is None:
            print(f"Warning: Could not read {image_name}. Skipping.")
            continue
            
        display_image = current_image.copy()
        current_boxes = []

        print("\n" + "="*50)
        print(f"Annotating: {image_name}")
        print("Instructions:")
        print(" - Draw boxes around text bubbles with your mouse.")
        print(" - Press 's' to SAVE and go to the next image.")
        print(" - Press 'r' to RESET all boxes for the current image.")
        print(" - Press 'q' to QUIT the tool.")
        print("="*50)

        while True:
            cv2.imshow('Annotation Tool', display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'): # Save
                if not current_boxes:
                    print("Warning: You haven't drawn any boxes. Save anyway? (y/n)")
                    confirm_key = cv2.waitKey(0) & 0xFF
                    if confirm_key != ord('y'):
                        continue

                label_filename = os.path.splitext(image_name)[0] + '.txt'
                label_path = os.path.join(OUTPUT_ANNOTATIONS_FOLDER, label_filename)
                
                with open(label_path, 'w') as f:
                    for (x, y, w, h) in current_boxes:
                        f.write(f"{x} {y} {w} {h}\n")

                # Also save the image with boxes drawn on it for review
                review_image_path = os.path.join(OUTPUT_ANNOTATIONS_FOLDER, image_name)
                cv2.imwrite(review_image_path, display_image)

                print(f"Saved annotation for {image_name}")
                annotated_files.append(image_name)
                save_progress(annotated_files)
                break # Move to next image

            elif key == ord('r'): # Reset
                print("Resetting boxes for this image.")
                current_boxes = []
                display_image = current_image.copy()

            elif key == ord('q'): # Quit
                print("Quitting annotation tool.")
                cv2.destroyAllWindows()
                return
    
    cv2.destroyAllWindows()
    print("\nFinished all available images!")


if __name__ == "__main__":
    start_annotation()
