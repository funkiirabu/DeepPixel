from PIL import Image
import os

def resize_images(input_folder, output_folder, size=(512, 512)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    count = 0
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)
            resized_image = image.resize(size)

            new_filename = f"{count:05}.png"
            output_path = os.path.join(output_folder, new_filename)

            resized_image.save(output_path)
            print(f"Resized {filename} and saved to {output_path}")

            count += 1

script_dir = os.path.dirname(os.path.realpath(__file__)) # Get the directory where the script is located
input_folder = os.path.join(script_dir, '../public/input') # Adjust the path as needed
output_folder = os.path.join(script_dir, '../public/output') # Adjust the path as needed
resize_images(input_folder, output_folder)
