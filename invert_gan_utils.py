import torchvision


def get_image(image_path, opts):
    image = torchvision.io.read_image(image_path)
    image = image.to(opts.device).unsqueeze(0)
    return image