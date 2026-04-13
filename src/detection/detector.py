# detector.py
# Component Detection using YOLOv8
import functools
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

@functools.lru_cache(maxsize=1)
def _get_model():
    return YOLO('yolov8n.pt')

model = _get_model()

# Mapping YOLO classes to Myntra Component Types
# This ensures the detector speaks the same language as your recommender.py
CATEGORY_MAP = {
    'person': 'User',
    'tie': 'Accessories',
    'handbag': 'Accessories',
    'backpack': 'Accessories',
    'umbrella': 'Accessories',
    'clock': 'Accessories',
    'shirt': 'Topwear',
    'suit': 'Topwear',
    't-shirt': 'Topwear',
    'dress': 'Topwear',
    'coat': 'Topwear',
    'jeans': 'Bottomwear',
    'trousers': 'Bottomwear',
    'shorts': 'Bottomwear',
    'skirt': 'Bottomwear',
    'shoes': 'Footwear',
    'boots': 'Footwear',
    'sneakers': 'Footwear'
}

def detect_clothing_components(image):
    """
    Analyzes an image and returns a list of unique clothing categories found.
    Args:
        image: PIL Image object or path to image
    Returns:
        list: Unique component_types (e.g., ['Topwear', 'Bottomwear'])
    """
    # Convert PIL to OpenCV format if necessary
    if isinstance(image, Image.Image):
        img_array = np.array(image)
        img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_cv2 = image

    # Run inference
    results = model(img_cv2, verbose=False)
    
    detected_categories = set()
    
    for r in results:
        for box in r.boxes:
            # Get class name from YOLO
            label = model.names[int(box.cls)].lower()
            conf = float(box.conf)
            
            # Filter by confidence and mapping
            if conf > 0.3:
                if label in CATEGORY_MAP:
                    detected_categories.add(CATEGORY_MAP[label])
                # Special handling for generic 'person' to suggest default sets
                elif label == 'person':
                    detected_categories.add('Topwear')
                    detected_categories.add('Bottomwear')

    # Remove 'User' from the final product category list
    if 'User' in detected_categories:
        detected_categories.remove('User')
        
    return list(detected_categories)

if __name__ == "__main__":
    # Test logic
    print("Testing YOLO Detection...")
    # Add a path to a local image to test:
    # results = detect_clothing_components("path/to/test_image.jpg")
    # print(f"Detected Categories: {results}")