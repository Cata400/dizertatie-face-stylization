import random
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shuffle dataset')
    
    parser.add_argument('content_path', type=str, help='Content image path')
    parser.add_argument('style_path', type=str, help='Style image path')
    parser.add_argument('seed', type=int, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    content_files = sorted(os.listdir(args.content_path))
    style_files = os.listdir(args.style_path)
    random.shuffle(style_files)
    
    if len(content_files) > len(style_files):
        style_files = style_files * (len(content_files) // len(style_files) + 1)
    
    style_files = style_files[:len(content_files)]
        
    with open('style_shuffled.txt', 'w') as f:
        for file in style_files:
            f.write(os.path.join(args.style_path, file) + '\n')