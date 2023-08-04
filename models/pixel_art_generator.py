import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from models.networks import define_G

class PixelArtGenerator:
    def __init__(self, checkpoint_path, input_nc=3, output_nc=3, ngf=64, netG='resnet_9blocks', norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
        # Define device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the checkpoint file
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Print the keys from the checkpoint
        print("Checkpoint keys:", checkpoint.keys())

        # Create the generator
        self.generator = define_G(input_nc, output_nc, ngf, netG, norm, use_dropout, init_type, init_gain, gpu_ids)
        self.generator.to(self.device)
        
        # Print the keys from the generator's state_dict
        print("Model keys:", self.generator.state_dict().keys())

        # Load the pre-trained weights
        self.generator.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.generator.eval()  # Set to evaluation mode

    def transform_image(self, input_image):
        # Convert input_image to a format suitable for the generator
        # Note: no need to open the image from a file path since it's already a PIL.Image object
        image = input_image.convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        # Pass the image through the generator
        with torch.no_grad():
            transformed_image_tensor = self.generator(image_tensor)

        # Transform the output tensor to a numpy array
        transformed_image_array = ((transformed_image_tensor.squeeze().cpu().numpy() + 1) / 2 * 255).astype(np.uint8).transpose(1, 2, 0)
        transformed_image = Image.fromarray(transformed_image_array)

        return transformed_image
