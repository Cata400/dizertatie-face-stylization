# Performs Style Transfer on all images from two directories
# The content directory is iterated in order, while the style directory is shuffled
# If the style directory is smaller than the content directory, it will be repeated
# If the style directory is larger than the content directory, the remaining images will be ignored
# The output is saved in the output directory
# The output files are named after the content and style images
# The script will create the output directory if it does not exist

# Usage: ./main_dataset.sh

# Parameters
CONTENT_DIR=../../Datasets/ffhq1k_random_slice_0.3
STYLE_DIR=../../Datasets/sketches/sketches_all_resized
OUTPUT_DIR=../../Results/Gatys_ffhq_sketches_1k_random_slice_0.3
RANDOM_SEED=42

# Create the output directory
mkdir -p $OUTPUT_DIR

# Shuffle the style directory
# I used python in order to shuffle consistently across algorithms
python shuffle_dataset.py $CONTENT_DIR $STYLE_DIR $RANDOM_SEED

# Get style image names in the shuffled order from the shuffled file
STYLE_IMAGES=($(cat style_shuffled.txt))

# Iterate over the content directory
INDEX=0
for CONTENT_IMAGE in $CONTENT_DIR/*; do
    STYLE_IMAGE=${STYLE_IMAGES[$INDEX]}
    CONTENT_IMAGE_NAME=$(basename $CONTENT_IMAGE)
    STYLE_IMAGE_NAME=$(basename $STYLE_IMAGE)

    echo $CONTENT_IMAGE_NAME $STYLE_IMAGE_NAME
    echo $CONTENT_IMAGE_NAME $STYLE_IMAGE_NAME
    echo $CONTENT_IMAGE_NAME $STYLE_IMAGE_NAME
    echo $CONTENT_IMAGE_NAME $STYLE_IMAGE_NAME
    echo $CONTENT_IMAGE_NAME $STYLE_IMAGE_NAME

    OUTPUT_IMAGE="$OUTPUT_DIR/${CONTENT_IMAGE_NAME%.*}+${STYLE_IMAGE_NAME%.*}.png"
    
    # Perform Style Transfer
    python main.py $CONTENT_IMAGE $STYLE_IMAGE $OUTPUT_IMAGE

    # For debugging, if index is equal to 2 break
    # if [ $INDEX -eq 2 ]; then
    #     break   
    # fi

    INDEX=$((INDEX + 1))
done
