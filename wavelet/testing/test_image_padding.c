
// gcc -o test_image_padding test_image_padding.c ../util/image_padding.c -lm
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../util/stb_image.h"
#include "../util/stb_image_write.h"
# include "../util/image_padding.h"

int main(int argc, char *argv[]) {

    char* image_dir = "../images/cartoon.png";
    char* output_dir = "../images/cartoon_padded.png";


    int width, height, channels;
    unsigned char *input = stbi_load(image_dir, &width, &height, &channels, 0);
    if (!input) {
        printf("Error: Could not load input image.\n");
        return 1;
    }

    int padded_width = next_power_of_2(width);
    int padded_height = next_power_of_2(height);
    unsigned char *output = (unsigned char *)malloc(padded_width * padded_height * channels);
    pad_image(input, width, height, channels, output);

    // Save padded image
    int success = stbi_write_png(output_dir, padded_width, padded_height, channels, output, padded_width * channels);
    if (!success) {
        printf("Error: Could not save output image.\n");
    }

    stbi_image_free(input);
    free(output);

    return 0;
}