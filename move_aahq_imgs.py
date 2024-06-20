import os
import shutil


src_dir = '../Datasets/aahq/aligned'
dst_dir = '../Datasets/aahq/aligned_used'
results_dir = '../Results/Gatys_ffhq_aahq_1k/'


for img in sorted(os.listdir(results_dir)):
    aahq_img_name = img.split('+')[1]
    shutil.copy2(os.path.join(src_dir, aahq_img_name), os.path.join(dst_dir, aahq_img_name))
    