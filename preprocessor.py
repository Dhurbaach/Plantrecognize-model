import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img

# Path to your main train directory
TRAIN_DIR = r"C:\Users\amrit\OneDrive\Desktop\Plant model\plant_dataset\test"

TARGET_COUNT = 10

# Augmentation generator
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Loop through each subdirectory (class)
for class_name in os.listdir(TRAIN_DIR):
    class_path = os.path.join(TRAIN_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(images)

    print(f"Processing '{class_name}' - {num_images} images")

    # Case 1: More than 10 → remove extra
    if num_images > TARGET_COUNT:
        images_to_remove = random.sample(images, num_images - TARGET_COUNT)
        for img_name in images_to_remove:
            os.remove(os.path.join(class_path, img_name))

        print(f"Removed {len(images_to_remove)} images")

    # Case 2: Less than 10 → augment
    elif num_images < TARGET_COUNT:
        needed = TARGET_COUNT - num_images
        i = 0

        while needed > 0:
            img_name = random.choice(images)
            img_path = os.path.join(class_path, img_name)

            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            # Generate augmented images
            for batch in datagen.flow(x, batch_size=1):
                new_img_name = f"aug_{i}_{img_name}"
                save_img(os.path.join(class_path, new_img_name), batch[0])

                i += 1
                needed -= 1

                if needed <= 0:
                    break

        print(f"Added {i} augmented images")

    else:
        print("Already has 10 images")

print("✅ Dataset balancing complete!")