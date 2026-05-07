from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import re
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def generate_blip_caption(image_path, prompt):
    image = Image.open(image_path).convert("RGB")

    text_prompt = prompt.strip() if prompt else "a photo of a plant"
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        # Increase length and sampling diversity. Generate multiple sampled sequences
        # and pick one at random to reduce deterministic repeats.
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=1,
            do_sample=True,
            num_return_sequences=3,
            temperature=0.9,
            top_p=0.95,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
        )

    # If multiple sequences returned, pick one randomly to add variety
    selected = outputs[torch.randint(len(outputs), (1,)).item()]
    caption = processor.tokenizer.decode(selected, skip_special_tokens=True).strip()

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

    if caption:
        caption = capitalize_sentences(caption)

    return caption