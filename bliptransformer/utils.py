import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PLANT_DESCRIPTIONS_PATH = BASE_DIR / "plant_descriptions_database.json"

with open(PLANT_DESCRIPTIONS_PATH, "r", encoding="utf-8") as f:
    plant_descriptions = json.load(f)


def build_prompt(plant):
    return (
        f"This is {plant['common_name']} ({plant['scientific_name']}). "
        f"It belongs to the {plant['family']} family. "
        f"{plant['description']} "
        f"It is used for {', '.join(plant['uses'])}."
    )

def get_plant_info(class_name):
    return plant_descriptions.get(class_name, None)