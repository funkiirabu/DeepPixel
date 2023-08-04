from models.pixel_art_generator import PixelArtGenerator

def main(input_image_path, output_image_path):
    # Path to the pre-trained generator checkpoint
    checkpoint_path = './checkpoints/pixel_cyclegan/latest_net_G_A.pth'

    # Parameters used during training (adjust these as needed)
    input_nc = 3
    output_nc = 3
    ngf = 64
    netG = 'resnet_9blocks'
    norm = 'instance'
    use_dropout = False
    init_type = 'normal'
    init_gain = 0.02
    gpu_ids = []

    # Create the PixelArtGenerator
    pixel_art_generator = PixelArtGenerator(checkpoint_path, input_nc, output_nc, ngf, netG, norm, use_dropout, init_type, init_gain, gpu_ids)

    # Use the transform_image method to create pixel art from an input image
    transformed_image = pixel_art_generator.transform_image(input_image_path, output_image_path)
    print(f"Transformed image saved to {output_image_path}")

if __name__ == '__main__':
    input_image_path = '../public/input/image1.png'
    output_image_path = '../public/output/image2.png'
    main(input_image_path, output_image_path)
