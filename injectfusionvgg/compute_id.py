import os
from losses import id_loss
from PIL import Image
from torchvision import transforms

# styles = {
#     'celeba': '/home/catalin/Desktop/Disertatie/Datasets/celeba_hq_lmdb/raw_images/test/images/',
#     'aahq': '/home/catalin/Desktop/Disertatie/Datasets/aahq/aligned_used/',
#     'sketches': '/home/catalin/Desktop/Disertatie/Datasets/sketches/sketches_all_resized/'
#     }

contents = {
    'ffhq': '/home/catalin/Desktop/Disertatie/Datasets/ffhq1k/'
}

results_dir = '../../Results/'
result_file = '../../Metrics/id_results.txt'
f = open(result_file, 'w')


def id_score(gen_path, content_path, id_model):
    gen_image = Image.open(gen_path)
    content_image = Image.open(content_path)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    gen_image = transform(gen_image)
    content_image = transform(content_image)
    
    gen_image = 2 * gen_image - 1
    content_image = 2 * content_image - 1
    
    gen_image = gen_image.unsqueeze(0).to('cuda')
    content_image = content_image.unsqueeze(0).to('cuda')
    
    id = 1 - id_model(content_image, gen_image)
    
    return id.item()
    


if __name__ == '__main__':
    id_loss_func = id_loss.IDLoss().to('cuda').eval()
    
    # gen_path = '/home/catalin/Desktop/Disertatie/dizertatie-face-stylization/ddim/experiment_vgg/image_samples/images_slerp_good/pred_0_projector_good_slerp_boost.png'
    # content_path = '/home/catalin/Desktop/Disertatie/Datasets/celeba_hq_lmdb/raw_images/test/images/098626.jpg'
    # id = id_score(gen_path, content_path, id_loss_func)
    # print(id)


    for directory in sorted(os.listdir(results_dir)):
        if 'ffhq' in directory and 'old' not in directory:
            # style = directory.split('_')[2]
            content = directory.split('_')[1]
            
            print(f'Calculating ID for {directory}...')
            print(f'Calculating ID for {directory}...', file=f)
            
            id_scores = []
            
            for i, image in enumerate(sorted(os.listdir(os.path.join(results_dir, directory)))):
                content_image = image.split('+')[0] + '.png'
                
                gen_path = os.path.join(results_dir, directory, image)
                content_path = os.path.join(contents[content], content_image)
                
                id = id_score(gen_path, content_path, id_loss_func)
                id_scores.append(id)
                
            avg_id = sum(id_scores) / len(id_scores)
            
            print(f'Average ID for {directory}: {avg_id}')
            print(f'Average ID for {directory}: {avg_id}', file=f)
            
            print()
            print("", file=f)