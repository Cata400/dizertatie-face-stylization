import os
from PIL import Image


if __name__ == '__main__':
    # Open an image file
    img_path = os.path.join('..', 'Images', 'JojoGAN', 'Aligned', 'arnold_aligned_aligned.jpg')
    save_path = os.path.join('..', 'Images', 'old', 'arnold_aligned_aligned_gray.jpg')
    
    img = Image.open(img_path)

    # Convert the image to grayscale
    gray_img = img.convert('L')

    # Save the grayscale image
    gray_img.save(save_path)