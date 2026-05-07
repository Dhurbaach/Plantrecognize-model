from bliptransformer.caption import generate_blip_caption
from bliptransformer.utils import get_plant_info, build_prompt


def generate_caption_for_plant(image_path: str, class_name: str) -> dict | None:
    """Generate caption for a plant given its image path and predicted class name.
    
    Args:
        image_path: Path to the plant image
        class_name: Predicted class name from the MobileNet model
    
    Returns:
        Dictionary with plant info and generated caption, or None if info unavailable
    """
    print(f"Predicted Class: {class_name}")
    class_name = class_name.replace(" ", "").strip()  # Remove spaces for matching
   
    # Step 1: Get JSON data for plant information
    plant_info = get_plant_info(class_name)

    if not plant_info:
        # fallback to raw BLIP (important safety)
        caption = generate_blip_caption(image_path, "Describe this plant")
        return {
            "class": class_name,
            "caption": caption,
            "note": "Fallback mode (no JSON match)"
        }

    # Step 2: Build prompt for BLIP transformer
    prompt = build_prompt(plant_info)

    # Step 3: Generate caption using BLIP transformer
    caption = generate_blip_caption(image_path, prompt)
    # final_caption=f"{caption} | Info: {prompt}"

    # Step 4: Return result with caption and plant metadata
    result = {
        "caption": caption,
        "scientific_name": plant_info.get("scientific_name", ""),
        "uses": plant_info.get("uses", [])
    }
    return result