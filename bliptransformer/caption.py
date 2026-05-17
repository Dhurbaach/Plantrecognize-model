from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import re
import torch
import os

# Define local paths for the model and processor
MODEL_PATH = "./blip_model/model"
PROCESSOR_PATH = "./blip_model/processor"

# Check if the model and processor are saved locally
if os.path.exists(MODEL_PATH) and os.path.exists(PROCESSOR_PATH):
    print("Loading BLIP model from local path...")
    processor = BlipProcessor.from_pretrained(PROCESSOR_PATH)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH)
else:
    print("Downloading and saving BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Create directories if they don't exist
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(PROCESSOR_PATH, exist_ok=True)
    
    # Save the model and processor locally
    model.save_pretrained(MODEL_PATH)
    processor.save_pretrained(PROCESSOR_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def _normalize_caption(text: str) -> str:
    if not text:
        return ""

    def capitalize_sentences(text: str) -> str:
        parts = re.split(r"([.!?]+\s*)", text)
        result = []

        for index in range(0, len(parts), 2):
            sentence = parts[index].strip()
            separator = parts[index + 1] if index + 1 < len(parts) else ""

            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                result.append(sentence)

            if separator:
                result.append(separator)

        return "".join(result).strip()
    text = re.sub(r"\s+", " ", text).strip()
    text = capitalize_sentences(text)

    # Ensure the caption ends with a complete sentence.
    last_punctuation = -1
    for p in ".!?":
        last_punctuation = max(last_punctuation, text.rfind(p))

    if last_punctuation != -1 and last_punctuation < len(text) - 1:
        text = text[:last_punctuation + 1]

    return text


def _long_hint(prompt: str | None) -> str | None:
    if not prompt:
        return None

    compact = re.sub(r"\s+", " ", prompt).strip()
    if not compact:
        return None

    # Keep enough prompt context, but cap it to reduce hallucinated copying.
    words = compact.split()
    return " ".join(words[:40])


def _are_similar(a: str, b: str) -> bool:
    if not a or not b:
        return False

    a_words = set(re.findall(r"[a-zA-Z]+", a.lower()))
    b_words = set(re.findall(r"[a-zA-Z]+", b.lower()))
    if not a_words or not b_words:
        return False

    overlap = len(a_words & b_words) / max(1, len(a_words | b_words))
    return overlap >= 0.65


def _generate_from_inputs(inputs: dict) -> str:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            num_beams=2,
            do_sample=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
        )

    return processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def generate_blip_caption(image_path, prompt=None):
    image = Image.open(image_path).convert("RGB")

    # 1) Always generate an image-grounded caption without text conditioning.
    visual_inputs = processor(images=image, return_tensors="pt")
    visual_inputs = {key: value.to(device) for key, value in visual_inputs.items()}
    visual_caption = _normalize_caption(_generate_from_inputs(visual_inputs))

    # 2) Also generate a prompt-guided caption to include contextual details.
    hint = _long_hint(prompt)
    if not hint:
        return visual_caption

    guided_inputs = processor(images=image, text=hint, return_tensors="pt")
    guided_inputs = {key: value.to(device) for key, value in guided_inputs.items()}
    guided_caption = _normalize_caption(_generate_from_inputs(guided_inputs))

    if not guided_caption:
        return visual_caption

    # Keep visual details first; append prompt-guided text only when distinct.
    if _are_similar(visual_caption, guided_caption):
        final_caption = visual_caption
    else:
        # Ensure visual caption ends with a period before appending more text.
        if not visual_caption.endswith((".", "!", "?")):
            visual_caption += "."
        final_caption = f"{visual_caption} {guided_caption}"

    # Prepend the desired starting phrase.
    if final_caption:
        return f"The uploaded image shows {final_caption[0].lower() + final_caption[1:]}"
    return ""