"""
Smart Diet AI - Gradio App for Hugging Face Spaces
Food Detection + Calorie Calculation API
Fixed for Gradio 6.0+
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
CSV_PATH = "data.csv"

# Physical assumptions
PLATE_DIAMETER_CM = 25.0
FOOD_DEPTH_CM = 2.0

# Detection settings
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
MAX_DETECTIONS = 20
IMG_SIZE = 640


# ============================================================
# LOAD RESOURCES
# ============================================================

print("🚀 Loading model and data...")

# Load model
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)
print(f"✅ Model loaded: {MODEL_PATH}")

# Load CSV
if not Path(CSV_PATH).exists():
    raise FileNotFoundError(f"❌ CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
df['food_name'] = df['food_name'].str.lower().str.strip()

# Clean numeric columns
numeric_cols = ['calories_100g', 'protein_100g', 'carbs_100g', 'fat_100g', 'density_cm']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['calories_100g', 'density_cm'])
df[['protein_100g', 'carbs_100g', 'fat_100g']] = df[['protein_100g', 'carbs_100g', 'fat_100g']].fillna(0)

print(f"✅ Nutrition data loaded: {len(df)} foods")

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Device: {DEVICE}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_nutrition(class_name, box_area_px, img_width_px):
    """Calculate weight and nutrition from bounding box"""
    
    food_name_clean = class_name.lower().strip()
    row = df[df['food_name'] == food_name_clean]
    
    if row.empty:
        row = df[df['food_name'].str.contains(food_name_clean, na=False, regex=False)]
    
    if row.empty:
        return {
            "weight": 0,
            "calories": 0,
            "protein": 0,
            "carbs": 0,
            "fat": 0,
            "found": False
        }
    
    data = row.iloc[0]
    cal_100g = float(data['calories_100g'])
    protein_100g = float(data['protein_100g'])
    carbs_100g = float(data['carbs_100g'])
    fat_100g = float(data['fat_100g'])
    density = float(data['density_cm'])
    
    # Calculate weight
    cm_per_pixel = PLATE_DIAMETER_CM / img_width_px
    area_cm2 = box_area_px * (cm_per_pixel ** 2)
    volume_cm3 = area_cm2 * FOOD_DEPTH_CM
    weight_g = volume_cm3 * density
    weight_g = max(10, min(weight_g, 500))
    
    # Calculate nutrition
    scale = weight_g / 100.0
    
    return {
        "weight": int(round(weight_g)),
        "calories": int(round(cal_100g * scale)),
        "protein": int(round(protein_100g * scale)),
        "carbs": int(round(carbs_100g * scale)),
        "fat": int(round(fat_100g * scale)),
        "found": True
    }


# ============================================================
# MAIN PREDICTION
# ============================================================

def predict_food(image):
    """Detect food and calculate nutrition"""
    
    if image is None:
        return None, {
            "error": "No image provided",
            "items": [],
            "totals": {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        }
    
    try:
        # Convert RGB to BGR
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_h, img_w = img_bgr.shape[:2]
        
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
        
        # Process results
        detections = []
        totals = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_area = (x2 - x1) * (y2 - y1)
                
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                confidence = float(box.conf[0])
                
                # Calculate nutrition
                nutrition = calculate_nutrition(cls_name, box_area, img_w)
                
                if nutrition['found']:
                    totals['calories'] += nutrition['calories']
                    totals['protein'] += nutrition['protein']
                    totals['carbs'] += nutrition['carbs']
                    totals['fat'] += nutrition['fat']
                
                # Draw on image
                color = (0, 255, 0) if nutrition['found'] else (0, 165, 255)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                
                label = f"{cls_name} {nutrition['calories']}kcal"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_bgr, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
                cv2.putText(img_bgr, label, (x1 + 5, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                detections.append({
                    "name": cls_name,
                    "confidence": round(confidence, 3),
                    "box": [x1, y1, x2, y2],
                    "weight_g": nutrition['weight'],
                    "calories": nutrition['calories'],
                    "macros": {
                        "protein": nutrition['protein'],
                        "carbs": nutrition['carbs'],
                        "fat": nutrition['fat']
                    },
                    "found": nutrition['found']
                })
        
        # Add summary banner
        if totals['calories'] > 0:
            banner = f"Total: {totals['calories']}kcal | P:{totals['protein']}g C:{totals['carbs']}g F:{totals['fat']}g"
            cv2.rectangle(img_bgr, (0, 0), (img_w, 40), (0, 0, 0), -1)
            cv2.putText(img_bgr, banner, (10, 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Convert back to RGB
        output_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # JSON result
        json_result = {
            "success": True,
            "total_items": len(detections),
            "totals": totals,
            "items": detections
        }
        
        return output_img, json_result
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return image, {
            "success": False,
            "error": str(e),
            "items": [],
            "totals": {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        }


# ============================================================
# GRADIO INTERFACE (FIXED FOR GRADIO 6.0+)
# ============================================================

demo = gr.Interface(
    fn=predict_food,
    inputs=gr.Image(type="numpy", label="📸 Upload Food Image"),
    outputs=[
        gr.Image(type="numpy", label="🍽️ Detected Food Items"),
        gr.JSON(label="📊 Nutrition Data (API Output)")
    ],
    title="🥗 Smart Diet AI",
    description=(
        "Upload a food image to detect items and calculate calories. "
        "Powered by YOLOv11 + Nutrition Database.\n\n"
        "**API Output:** JSON format ready for mobile/web backend integration."
    ),
    article=(
        "### 📖 How It Works\n"
        "1. Upload an image of your meal\n"
        "2. AI detects food items using YOLOv11\n"
        "3. Calculates weight based on bounding box size\n"
        "4. Returns calories and macros (Protein, Carbs, Fat)\n\n"
        "### 🔧 API Integration\n"
        "The JSON output can be directly consumed by your backend.\n\n"
        f"### ⚙️ Settings\n"
        f"- Confidence: {CONFIDENCE_THRESHOLD}\n"
        f"- Plate Diameter: {PLATE_DIAMETER_CM} cm\n"
        f"- Device: {DEVICE.upper()}\n"
    ),
    # ✅ REMOVED deprecated parameters:
    # - analytics_enabled (removed in Gradio 6.0)
    # - allow_flagging (removed in Gradio 6.0)
)


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch()