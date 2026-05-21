"""
Run as an API:
    python predict.py

POST /predict with multipart/form-data:
    - image: image file
    - organ: optional text field

Optional CLI test:
    python predict.py path/to/image.jpg
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from bliptransformer.main import generate_caption_for_plant

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_mobilenet_model.keras"
CONVNEXT_MODEL_PATH = BASE_DIR / "best_convnext.pth"
CLASS_INDICES_PATH = BASE_DIR / "class_indices.json"

app = FastAPI()


def load_class_names() -> list[str]:
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        try:
            return [data[str(index)] for index in range(len(data))]
        except KeyError:
            return [value for _, value in sorted(data.items(), key=lambda item: int(item[0]))]

    raise ValueError("class_indices.json must contain a list or an index-to-name mapping")


def load_model_and_labels() -> tuple[tf.keras.Model, torch.nn.Module, list[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not CONVNEXT_MODEL_PATH.exists():
        raise FileNotFoundError(f"ConvNeXt model file not found: {CONVNEXT_MODEL_PATH}")
    if not CLASS_INDICES_PATH.exists():
        raise FileNotFoundError(f"Class index file not found: {CLASS_INDICES_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    # Attempt to load ConvNeXt model. The .pth file may contain a full model
    # or only a state_dict. Handle both cases and try common ConvNeXt variants
    # using timm when only a state_dict is present.
    loaded = torch.load(CONVNEXT_MODEL_PATH, map_location=torch.device('cpu'))
    class_names = load_class_names()

    # If the file already contains an nn.Module, use it directly.
    if isinstance(loaded, torch.nn.Module):
        convnext_model = loaded
        return model, convnext_model, class_names

    # Otherwise expect a state dict or a checkpoint containing a state dict.
    if isinstance(loaded, dict):
        # Common checkpoint wrappers
        state_dict = None
        for key in ("state_dict", "model_state_dict", "model", "model_state"):
            if key in loaded:
                state_dict = loaded[key]
                break

        # If not a wrapped checkpoint, assume the dict itself is the state dict
        if state_dict is None:
            state_dict = loaded

        # Remove potential DataParallel prefixes
        cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}

        try:
            import timm
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "timm is required to reconstruct ConvNeXt from a state_dict. "
                "Install it with `pip install timm` or provide a pickled nn.Module."
            ) from exc

        num_classes = len(class_names)
        convnext_model = None
        variants = ["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"]
        last_err = None
        for variant in variants:
            try:
                candidate = timm.create_model(variant, pretrained=False, num_classes=num_classes)
                candidate.load_state_dict(cleaned_state)
                candidate.eval()
                convnext_model = candidate
                break
            except Exception as e:  # try next variant
                last_err = e

        if convnext_model is None:
            raise RuntimeError(
                "Failed to load ConvNeXt state_dict into a known architecture. "
                "Either save the full nn.Module (torch.save(model)) or install the exact model code used when saving. "
                f"Last error: {last_err}"
            )

        return model, convnext_model, class_names

    raise RuntimeError("Unsupported ConvNeXt model file format")


MODEL, CONVNEXT_MODEL, CLASS_NAMES = load_model_and_labels()


def predict_file(file_path: str) -> dict:
    img = keras_image.load_img(file_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Standard ImageNet normalization for ConvNeXt
    img_normalized = img_array / 255.0
    
    # Normalize with ImageNet statistics
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_normalized - mean) / std
    
    try:
        with torch.no_grad():
            # Convert to tensor and permute from (batch, height, width, channels) to (batch, channels, height, width)
            img_tensor = torch.from_numpy(img_normalized).permute(0, 3, 1, 2).float()
            convnext_predictions = CONVNEXT_MODEL(img_tensor)
            
            # Handle both tensor and dict outputs (some models return dicts)
            if isinstance(convnext_predictions, dict):
                convnext_predictions_np = convnext_predictions.get('logits', convnext_predictions.get('output', convnext_predictions)).detach().cpu().numpy()
            else:
                convnext_predictions_np = convnext_predictions.detach().cpu().numpy()
        
        # Convert model outputs (logits) to probabilities using softmax so
        # confidence is a proper probability between 0 and 1.
        exp_scores = np.exp(convnext_predictions_np - np.max(convnext_predictions_np, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        predicted_index = int(np.argmax(probs, axis=1)[0])
        confidence = float(np.max(probs))
        top_indices = np.argsort(probs[0])[::-1][:5]

        plant_name = CLASS_NAMES[predicted_index]
        if confidence <= 0.25:
            plant_name = "none"

        return {
            "success": True,
            "message": "Plant identified successfully",
            "file_path": file_path,
            "plant_name": plant_name,
            "class_index": predicted_index,
            "confidence": confidence,
            "confidence_percentage": round(confidence * 100, 2),
            "top_5_predictions": {
                CLASS_NAMES[i]: float(probs[0][i]) for i in top_indices
            },
        }
    except Exception as e:
        import traceback
        raise Exception(f"Error during ConvNeXt prediction: {str(e)}\n{traceback.format_exc()}")


@app.get("/health")
async def health() -> dict:
    return {"success": True, "message": "Plant model service is running"}


@app.post("/predict")
async def predict(image: UploadFile = File(...), organ: Optional[str] = Form(None)) -> JSONResponse:
    if not image.filename:
        raise HTTPException(status_code=400, detail="Image file is required")

    temp_path = BASE_DIR / f"_upload_{image.filename}"

    try:
        content = await image.read()
        temp_path.write_bytes(content)

        # First, predict the class using ConvNeXt in a threadpool
        result = await run_in_threadpool(predict_file, str(temp_path))

        # Then, generate caption using BLIP and get plant metadata (run in threadpool)
        from bliptransformer.utils import get_plant_info

        class_name = result["plant_name"].replace(" ", "").strip()

        # If confidence is low, generate a generic caption and get info for "none" class.
        # Otherwise, use the predicted class name.
        if result["confidence"] <= 0.25:
            class_name_for_info = "none"
            caption_result = await run_in_threadpool(generate_caption_for_plant, str(temp_path), class_name=class_name_for_info)
        else:
            class_name_for_info = class_name
            caption_result = await run_in_threadpool(generate_caption_for_plant, str(temp_path), class_name)

        if caption_result:
            result.update(caption_result)

        plant_info = await run_in_threadpool(get_plant_info, class_name_for_info)
        if plant_info:
            result["common_name"] = plant_info.get("common_name", "")
            result["nepali_name"] = plant_info.get("local_nepali_name", "")
            result["scientific_name"] = plant_info.get("scientific_name", "")
            result["uses"] = plant_info.get("uses", [])

        return JSONResponse(content=result, status_code=200)
    except Exception as exc:
        return JSONResponse(content={"success": False, "message": "Error identifying plant", "error": str(exc)}, status_code=500)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
