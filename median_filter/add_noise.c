/*
    Description: Adds Gaussian or Salt & Pepper noise to a PNG image.
    Compile: gcc -o add_noise add_noise.c -lpng
    Excute: ./add_noise ./data/org/color_org.png ./data/noisy/color_saltpepper.png 2
*/

#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <time.h>

void add_gaussian_noise(png_bytep *image, int width, int height, int channels, double mean, double stddev) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                double noise = stddev * ((double)rand() / RAND_MAX) + mean;
                int new_value = image[y][x * channels + c] + (int)noise;
                image[y][x * channels + c] = (new_value > 255) ? 255 : (new_value < 0) ? 0 : new_value;
            }
        }
    }
}

void add_salt_and_pepper_noise(png_bytep *image, int width, int height, int channels, float percentage) {
    int num_pixels_to_change = (int)(percentage * width * height);

    for (int i = 0; i < num_pixels_to_change; ++i) {
        int row = rand() % height;
        int col = rand() % width;
        int value = (rand() % 2 == 0) ? 0 : 255;

        for (int c = 0; c < channels; c++) {
            image[row][col * channels + c] = value;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <input_file> <output_file> <noise_type>\n", argv[0]);
        printf("noise_type: 1 for Gaussian, 2 for Salt & Pepper\n");
        return 1;
    }

    srand(time(NULL));

    char *input_filename = argv[1];
    char *output_filename = argv[2];
    int noise_type = atoi(argv[3]);

    // Read the input image
    FILE *input_file = fopen(input_filename, "rb");
    if (!input_file) {
        fprintf(stderr, "Error: Unable to read input file: %s\n", input_filename);
        return 1;
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, input_file);
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_EXPAND, NULL);

    int width = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);

    int channels;
    if (color_type == PNG_COLOR_TYPE_GRAY) {
        channels = 1;
    } else if (color_type == PNG_COLOR_TYPE_RGB) {
        channels = 3;
    } else {
        fprintf(stderr, "Error: The input image must be either a grayscale or an RGB image.\n");
        return 1;
    }

    png_bytep *row_pointers = png_get_rows(png_ptr, info_ptr);

    if (noise_type == 1) {
        add_gaussian_noise(row_pointers, width, height, channels, 0, 25);
    } else if (noise_type == 2) {
        add_salt_and_pepper_noise(row_pointers, width, height, channels, 0.05);
    } else {
        fprintf(stderr, "Error: Invalid noise type\n");
        return 1;
    }

    fclose(input_file);

    // Write the output image
    FILE *output_file = fopen(output_filename, "wb");
    if (!output_file) {
        fprintf(stderr, "Error: Unable to write output file: %s\n", output_filename);
        return 1;
    }

    png_structp write_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop write_info_ptr = png_create_info_struct(write_ptr);
    png_init_io(write_ptr, output_file);

    png_set_IHDR(write_ptr, write_info_ptr, width, height, bit_depth, color_type,
                PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_rows(write_ptr, write_info_ptr, row_pointers);
    png_write_png(write_ptr, write_info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    png_write_end(write_ptr, write_info_ptr);

    fclose(output_file);

    // Clean up
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    png_destroy_write_struct(&write_ptr, &write_info_ptr);

    printf("Success: Noise added and new image saved to: %s\n", output_filename);
    return 0;
}

