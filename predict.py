# !/usr/bin/env python3
"""Local plant identification inference service.

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
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_mobilenet_model.keras"
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


def load_model_and_labels() -> tuple[tf.keras.Model, list[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not CLASS_INDICES_PATH.exists():
        raise FileNotFoundError(f"Class index file not found: {CLASS_INDICES_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names()
    return model, class_names


MODEL, CLASS_NAMES = load_model_and_labels()


def predict_file(file_path: str) -> dict:
    img = keras_image.load_img(file_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = MODEL.predict(img_array, verbose=0)
    predicted_index = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions))
    top_indices = np.argsort(predictions[0])[::-1][:5]

    return {
        "success": True,
        "message": "Plant identified successfully",
        "file_path": file_path,
        "plant_name": CLASS_NAMES[predicted_index],
        "class_index": predicted_index,
        "confidence": confidence,
        "confidence_percentage": round(confidence * 100, 2),
        "top_5_predictions": {
            CLASS_NAMES[i]: float(predictions[0][i]) for i in top_indices
        },
    }


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

        # First, predict the class using MobileNet in a threadpool
        result = await run_in_threadpool(predict_file, str(temp_path))

        # Then, generate caption using BLIP and get plant metadata (run in threadpool)
        from bliptransformer.main import generate_caption_for_plant
        from bliptransformer.utils import get_plant_info

        class_name = result["plant_name"].replace(" ", "").strip()
        caption_result = await run_in_threadpool(generate_caption_for_plant, str(temp_path), class_name)

        if caption_result:
            result.update(caption_result)

        plant_info = await run_in_threadpool(get_plant_info, class_name)
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
