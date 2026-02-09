"""
Smart Diet AI - Final Version
Shows nutrition per 100g as text + annotated image
JSON hidden in accordion (for backend developers only)
"""

import gradio as gr
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "best.pt"
DATA_PATH = "data.csv"

# Detection settings
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
MAX_DETECTIONS = 20
IMG_SIZE = 640


# ============================================================
# LOAD RESOURCES
# ============================================================

print("🚀 Loading model and data...")

# Load YOLO model
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)
print(f"✅ Model loaded: {MODEL_PATH}")

# Load nutrition database
if not Path(DATA_PATH).exists():
    raise FileNotFoundError(f"❌ Data file not found: {DATA_PATH}")

try:
    df = pd.read_csv(DATA_PATH)
    df['food_name'] = df['food_name'].str.lower().str.strip()
    
    numeric_cols = ['calories_100g', 'protein_100g', 'carbs_100g', 'fat_100g']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['calories_100g'])
    df[['protein_100g', 'carbs_100g', 'fat_100g']] = \
        df[['protein_100g', 'carbs_100g', 'fat_100g']].fillna(0)
    
    print(f"✅ Nutrition data loaded: {len(df)} foods")
    
except Exception as e:
    raise RuntimeError(f"❌ Failed to load data: {e}")

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Device: {DEVICE}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_food_info(class_name):
    """Get nutrition info per 100g and description"""
    
    food_name_clean = class_name.lower().strip()
    
    # Try exact match
    row = df[df['food_name'] == food_name_clean]
    
    # Try partial match
    if row.empty:
        row = df[df['food_name'].str.contains(food_name_clean, na=False, regex=False)]
    
    # Not found
    if row.empty:
        return {
            "calories_100g": 0,
            "protein_100g": 0,
            "carbs_100g": 0,
            "fat_100g": 0,
            "info": "No information available for this food item.",
            "found": False
        }
    
    # Extract data
    data = row.iloc[0]
    
    return {
        "calories_100g": int(data['calories_100g']),
        "protein_100g": round(float(data['protein_100g']), 1),
        "carbs_100g": round(float(data['carbs_100g']), 1),
        "fat_100g": round(float(data['fat_100g']), 1),
        "info": str(data['info']) if 'info' in data and pd.notna(data['info']) else "No additional information.",
        "found": True
    }


def draw_annotations(image, detections):
    """Draw bounding boxes and labels"""
    
    output = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        name = det['name']
        calories = det['calories_100g']
        confidence = det['confidence']
        found = det.get('found', True)
        
        # Color
        color = (0, 255, 0) if found else (0, 165, 255)
        
        # Draw box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f"{name} ({calories}kcal/100g)"
        
        # Label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
        
        # Label text
        cv2.putText(output, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Confidence
        conf_text = f"{confidence:.0%}"
        cv2.putText(output, conf_text, (x1, y2 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Convert BGR to RGB
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


def format_text_output(detections):
    """Create formatted text output with nutrition info"""
    
    if not detections:
        return "❌ No food items detected.\n\nTry uploading a clearer image with visible food items."
    
    # Header
    output = "=" * 60 + "\n"
    output += "🍽️  DETECTED FOOD ITEMS - NUTRITION PER 100g\n"
    output += "=" * 60 + "\n\n"
    
    # Each item
    for i, det in enumerate(detections, 1):
        output += f"┌─ [{i}] {det['name'].upper()}\n"
        output += f"│\n"
        
        if det['found']:
            output += f"│  📊 Nutrition per 100g:\n"
            output += f"│  ├─ 🔥 Calories : {det['calories_100g']} kcal\n"
            output += f"│  ├─ 💪 Protein  : {det['protein_100g']} g\n"
            output += f"│  ├─ 🍞 Carbs    : {det['carbs_100g']} g\n"
            output += f"│  └─ 🥑 Fat      : {det['fat_100g']} g\n"
            output += f"│\n"
            output += f"│  ℹ️  Information:\n"
            
            # Wrap info text
            info_text = det['info']
            max_width = 55
            words = info_text.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= max_width:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_length = len(word)
            
            if current_line:
                lines.append(" ".join(current_line))
            
            for line in lines:
                output += f"│     {line}\n"
            
            output += f"│\n"
            output += f"│  🎯 Confidence : {det['confidence']:.1%}\n"
        else:
            output += f"│  ⚠️  Not found in database\n"
            output += f"│  🎯 Confidence : {det['confidence']:.1%}\n"
        
        output += f"└─" + "─" * 58 + "\n\n"
    
    # Summary
    output += "=" * 60 + "\n"
    output += f"📌 SUMMARY\n"
    output += "=" * 60 + "\n"
    output += f"Total items detected: {len(detections)}\n"
    
    found_count = sum(1 for d in detections if d['found'])
    output += f"Items in database: {found_count}/{len(detections)}\n"
    
    output += "\n💡 Note: All values are per 100 grams.\n"
    output += "   Multiply by your actual portion size to get total nutrition.\n"
    output += "   Example: 150g × values shown = actual intake\n"
    
    return output


# ============================================================
# MAIN PREDICTION
# ============================================================

def predict_food(image):
    """
    Detect food and return nutrition per 100g
    
    Returns:
        tuple: (annotated_image, text_output, json_result)
    """
    
    if image is None:
        return (
            None,
            "❌ No image provided. Please upload an image.",
            {"error": "No image provided", "items": []}
        )
    
    try:
        # Convert RGB to BGR
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Run detection
        results = model.predict(
            img_bgr,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            max_det=MAX_DETECTIONS,
            device=DEVICE,
            verbose=False
        )
        
        # Process detections
        detections = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                confidence = float(box.conf[0])
                
                # Get nutrition info
                food_data = get_food_info(cls_name)
                
                detection = {
                    "name": cls_name.title(),
                    "confidence": round(confidence, 3),
                    "box": [x1, y1, x2, y2],
                    "calories_100g": food_data['calories_100g'],
                    "protein_100g": food_data['protein_100g'],
                    "carbs_100g": food_data['carbs_100g'],
                    "fat_100g": food_data['fat_100g'],
                    "info": food_data['info'],
                    "found": food_data['found']
                }
                
                detections.append(detection)
        
        # Create outputs
        output_img = draw_annotations(img_bgr, detections)
        text_output = format_text_output(detections)
        
        json_result = {
            "success": True,
            "total_items": len(detections),
            "items": detections,
            "note": "All nutrition values are per 100 grams of food"
        }
        
        return output_img, text_output, json_result
    
    except Exception as e:
        error_text = f"❌ Error during prediction:\n\n{str(e)}\n\nPlease try again with a different image."
        return (
            image,
            error_text,
            {"success": False, "error": str(e), "items": []}
        )


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(theme=gr.themes.Soft(), title="Smart Diet AI") as demo:
    
    # Header
    gr.Markdown(
        """
        # 🥗 Smart Diet AI - Nutrition Database
        
        Upload a food image to detect items and get detailed nutrition information.
        
        **All nutritional values are shown per 100 grams** - multiply by your portion size for actual intake.
        """
    )
    
    with gr.Row():
        # Input column
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="numpy",
                label="📸 Upload Food Image"
            )
            
            predict_btn = gr.Button(
                "🔍 Analyze Food",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown(
                """
                ### 💡 Tips:
                - Use clear, well-lit photos
                - Show food items clearly
                - Avoid blurry or dark images
                """
            )
        
        # Output column
        with gr.Column(scale=1):
            image_output = gr.Image(
                type="numpy",
                label="🍽️ Detected Food Items"
            )
    
    # Text output (full width)
    text_output = gr.Textbox(
        label="📊 Nutrition Information (Per 100g)",
        lines=20,
        max_lines=30,
        # show_copy_button=True
    )
    
    # JSON output (COLLAPSED by default - only for backend developers)
    with gr.Accordion("🔧 API Response (For Developers)", open=False):
        gr.Markdown(
            """
            **For Backend Integration Only**
            
            This JSON output is designed for programmatic access via API.
            Regular users don't need to see this - the text output above is more readable.
            """
        )
        json_output = gr.JSON(label="JSON Output")
    
    # Info section
    gr.Markdown(
        """
        ---
        ### 📖 How to Use
        
        1. **Upload** an image of your meal
        2. **Click** "Analyze Food" button
        3. **View** detected items with nutrition per 100g
        4. **Calculate** actual intake: Multiply values by your portion size
        
        **Example:** If you ate 150g of pizza and it shows 266 kcal/100g:
        - Actual calories = 150g × 2.66 = **399 kcal**
        
        ### ⚙️ Current Settings
        
        - **Confidence Threshold:** 0.25
        - **Device:** GPU (if available)
        - **Database:** 99 foods with detailed information
        """
    )
    
    # Button action
    predict_btn.click(
        fn=predict_food,
        inputs=image_input,
        outputs=[image_output, text_output, json_output]
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch()