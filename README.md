# DeepPixel

## Overview
Deep Pixel is a deep learning project that uses a pre-trained model to transform regular images into pixel art. It utilizes a trained generator from a CycleGAN architecture with a specific design (Resnet with 9 blocks) and custom settings to achieve the transformation.

## Features
- **Transform Images to Pixel Art**: Feed in any standard image, and the model will generate a corresponding pixel art version.
- **Pre-trained Model**: Utilizes a pre-trained generator, making it fast and easy to use without needing to train the model from scratch.
- **Customizable Settings**: Allows customization of various parameters like the number of generator features, normalization type, dropout usage, initialization type, and gain.

## Usage
### Prerequisites
- Python 3.x
- PyTorch
- PIL
- torchvision

### Running the Generator
1. **Set Up the Environment**: Make sure you have the required libraries installed.
2. **Prepare Your Images**: Place the images you want to transform in a known directory.
3. **Run the Script**: Use the `generate_pixel_art.py` script to transform your images. You can customize the parameters as needed.

   ```bash
   $ python generate_pixel_art.py
   ```
## Model Architecture
The generator model is built using a Resnet architecture with 9 blocks. It's a part of the CycleGAN system, trained with specific parameters that suit the pixel art transformation task. The architecture includes several convolutional layers, normalization layers, and dropout layers to enable the transformation from standard images to pixel art.

## Acknowledgements
Special thanks to the creators of CycleGAN and contributors to the related libraries and tools that made this project possible. The trained model leverages state-of-the-art techniques in GANs to achieve impressive results in image transformation.

## License
This project is licensed under the MIT License. See the LICENSE file for details. Feel free to use, modify, and distribute the code according to the terms of the license.