"""
Smart Diet AI - Gradio App for Hugging Face Spaces
Food Detection + Calorie Calculation API
Optimized for deployment and backend integration
"""

import gradio as gr
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch
import os

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "best.pt"
CSV_PATH = "data.csv"

# Physical assumptions for calorie calculation
PLATE_DIAMETER_CM = 25.0
FOOD_DEPTH_CM = 2.0

# Detection settings
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
MAX_DETECTIONS = 20
IMG_SIZE = 640

# ============================================================
# LOAD RESOURCES (once at startup)
# ============================================================

print("🚀 Loading model and data...")

# Check if files exist
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
if not Path(CSV_PATH).exists():
    raise FileNotFoundError(f"❌ CSV not found: {CSV_PATH}")

# Load YOLO model
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# Load nutrition database
try:
    df = pd.read_csv(CSV_PATH)

    # Clean and validate data
    df['food_name'] = df['food_name'].str.lower().str.strip()

    # Ensure numeric columns
    numeric_cols = ['calories_100g', 'protein_100g', 'carbs_100g', 'fat_100g', 'density_cm']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows with missing critical data
    df = df.dropna(subset=['calories_100g', 'density_cm'])

    # Fill missing macros with 0
    df[['protein_100g', 'carbs_100g', 'fat_100g']] = \
        df[['protein_100g', 'carbs_100g', 'fat_100g']].fillna(0)

    print(f"✅ Nutrition data loaded: {len(df)} foods")

except Exception as e:
    raise RuntimeError(f"❌ Failed to load CSV: {e}")

# Check GPU availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Device: {DEVICE}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_nutrition(class_name, box_area_px, img_width_px):
    """
    Calculate weight and nutrition from bounding box

    Returns:
        dict with weight, calories, macros
    """

    # Search for food in database (case-insensitive)
    food_name_clean = class_name.lower().strip()
    row = df[df['food_name'] == food_name_clean]

    # Try partial match if exact match fails
    if row.empty:
        row = df[df['food_name'].str.contains(food_name_clean, na=False, regex=False)]

    # Default values if not found
    if row.empty:
        return {
            "weight": 0,
            "calories": 0,
            "protein": 0,
            "carbs": 0,
            "fat": 0,
            "found": False
        }

    # Extract nutrition data (per 100g)
    data = row.iloc[0]
    cal_100g = float(data['calories_100g'])
    protein_100g = float(data['protein_100g'])
    carbs_100g = float(data['carbs_100g'])
    fat_100g = float(data['fat_100g'])
    density = float(data['density_cm'])

    # ── CALCULATE WEIGHT ──────────────────────────────────────
    # Convert pixels to cm
    cm_per_pixel = PLATE_DIAMETER_CM / img_width_px

    # Calculate area in cm²
    area_cm2 = box_area_px * (cm_per_pixel ** 2)

    # Calculate volume (area × depth)
    volume_cm3 = area_cm2 * FOOD_DEPTH_CM

    # Calculate weight (volume × density)
    weight_g = volume_cm3 * density

    # Apply reasonable limits
    weight_g = max(10, min(weight_g, 500))  # 10g to 500g

    # ── CALCULATE NUTRITION ───────────────────────────────────
    scale = weight_g / 100.0

    return {
        "weight": int(round(weight_g)),
        "calories": int(round(cal_100g * scale)),
        "protein": int(round(protein_100g * scale)),
        "carbs": int(round(carbs_100g * scale)),
        "fat": int(round(fat_100g * scale)),
        "found": True
    }


def draw_annotations(image, detections):
    """
    Draw bounding boxes and labels on image

    Args:
        image: numpy array (BGR)
        detections: list of detection dicts

    Returns:
        annotated image (RGB for Gradio)
    """

    output = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det['box']
        name = det['name']
        calories = det['calories']
        confidence = det['confidence']
        found = det.get('found', True)

        # Color: green if found in DB, orange if not
        color = (0, 255, 0) if found else (0, 165, 255)

        # Draw box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # Prepare label
        label = f"{name} {calories}kcal ({confidence:.0%})"

        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)

        # Draw label text
        cv2.putText(output, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Convert BGR to RGB for Gradio
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


def create_summary_banner(image, totals):
    """
    Add summary banner at top of image
    """

    img_h, img_w = image.shape[:2]

    # Banner text
    banner_text = (
        f"Total: {totals['calories']}kcal | "
        f"P:{totals['protein']}g C:{totals['carbs']}g F:{totals['fat']}g"
    )

    # Draw black background
    cv2.rectangle(image, (0, 0), (img_w, 40), (0, 0, 0), -1)

    # Draw text
    cv2.putText(image, banner_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return image


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_food(image):
    """
    Main function: detect food and calculate nutrition

    Args:
        image: numpy array from Gradio (RGB)

    Returns:
        tuple: (annotated_image, json_result)
    """

    # ── INPUT VALIDATION ──────────────────────────────────────
    if image is None:
        return None, {
            "error": "No image provided",
            "items": [],
            "totals": {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        }

    try:
        # Convert RGB to BGR for OpenCV/YOLO
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_h, img_w = img_bgr.shape[:2]

        # ── RUN DETECTION ─────────────────────────────────────────
        results = model.predict(
            img_bgr,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            max_det=MAX_DETECTIONS,
            device=DEVICE,
            verbose=False
        )

        # ── PROCESS RESULTS ───────────────────────────────────────
        detections = []
        totals = {
            "calories": 0,
            "protein": 0,
            "carbs": 0,
            "fat": 0
        }

        for result in results:
            for box in result.boxes:
                # Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height

                # Class info
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                confidence = float(box.conf[0])

                # Calculate nutrition
                nutrition = calculate_nutrition(cls_name, box_area, img_w)

                # Accumulate totals (only if found in DB)
                if nutrition['found']:
                    totals['calories'] += nutrition['calories']
                    totals['protein'] += nutrition['protein']
                    totals['carbs'] += nutrition['carbs']
                    totals['fat'] += nutrition['fat']

                # Add to detections list
                detection = {
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
                }

                detections.append(detection)

        # ── CREATE OUTPUT IMAGE ───────────────────────────────────
        output_img = draw_annotations(img_bgr, detections)

        # Add summary banner if any detections
        if totals['calories'] > 0:
            output_img = create_summary_banner(output_img, totals)

        # ── CREATE JSON RESPONSE ──────────────────────────────────
        json_result = {
            "success": True,
            "total_items": len(detections),
            "totals": totals,
            "items": detections
        }

        return output_img, json_result

    except Exception as e:
        # Error handling
        print(f"❌ Error during prediction: {e}")
        return image, {
            "success": False,
            "error": str(e),
            "items": [],
            "totals": {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        }


# ============================================================
# GRADIO INTERFACE
# ============================================================

# Custom CSS for better appearance
custom_css = """
#component-0 {
    max-width: 1200px;
    margin: auto;
}
.footer {
    text-align: center;
    margin-top: 20px;
}
"""

# Examples (optional - add sample images if available)
examples = [
    # ["example_plate.jpg"],  # uncomment and add your example images
]

# Create interface
demo = gr.Interface(
    fn=predict_food,

    # Inputs
    inputs=gr.Image(
        type="numpy",
        label="📸 Upload Food Image",
        sources=["upload", "webcam"],
    ),

    # Outputs
    outputs=[
        gr.Image(
            type="numpy",
            label="🍽️ Detected Food Items"
        ),
        gr.JSON(
            label="📊 Nutrition Data (API Output)"
        )
    ],

    # Metadata
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
        "The JSON output can be directly consumed by your backend:\n"
        "- `totals`: Overall nutritional values\n"
        "- `items`: Individual food items with details\n"
        "- `success`: API call status\n\n"
        "### ⚙️ Settings\n"
        f"- Detection Confidence: {CONFIDENCE_THRESHOLD}\n"
        f"- Plate Diameter: {PLATE_DIAMETER_CM} cm\n"
        f"- Device: {DEVICE.upper()}\n"
    ),

    # UI settings
    examples=examples if examples else None,
    cache_examples=False,
    css=custom_css,
    theme=gr.themes.Soft(),

    # API settings
    allow_flagging="never",
    analytics_enabled=False,
)

# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Allow external access (HF Spaces)
        server_port=7860,  # Default HF Spaces port
        share=False,  # HF handles sharing
        show_error=True,
    )