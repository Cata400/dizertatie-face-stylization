# dizertatie-face-stylization
This repository contains the code for my Master's thesis on face stylization using deep learning techniques. The project focuses on generating artistic styles for facial images while preserving identity features. 

For this project I implemented three different approaches:
- **Gatys**: A neural style transfer method that uses convolutional neural networks to apply artistic styles to images. It leverages the features extracted from pre-trained networks to blend content and style. It is based on the work of Gatys et al. and is known for its ability to create high-quality stylized images.

- **JojoGAN**: JojoGAN is a one-shot facial style transfer technique that fine-tunes a pre-trained StyleGAN2 model to apply the unique visual characteristics of a single reference image onto a target face while preserving identity. Designed for ease and flexibility, it creates a paired training dataset on-the-fly and quickly adapts the generator to new styles, achieving photorealistic or stylized outputs without the need for extensive datasets or retraining from scratch.

- **InjectFusionVGG**: InjectFusionVGG is a training-free style transfer method that combines the content of one image with the style of another using a VGG-based feature projector and a diffusion model. It operates by injecting content features directly into the denoising process, guiding the generation toward preserving structural details while applying the desired style. This approach enables fast and flexible image stylization without the need for fine-tuning or large-scale training.