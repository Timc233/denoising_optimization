#include "haar_wavelet.h"

void haar_1d(double* data, int length) {
    double* temp = (double*)malloc(length * sizeof(double));

    int h = length / 2;
    for (int i = 0; i < h; i++) {
        double sum = (data[2 * i] + data[2 * i + 1]) / sqrt(2);
        double difference = (data[2 * i] - data[2 * i + 1]) / sqrt(2);

        temp[i] = sum;
        temp[i + h] = difference;
    }

    for (int i = 0; i < length; i++) {
        data[i] = temp[i];
    }

    free(temp);
}

void inverse_haar_1d(double* data, int length) {
    double* temp = (double*)malloc(length * sizeof(double));

    int h = length / 2;
    for (int i = 0; i < h; i++) {
        double sum = data[i];
        double difference = data[i + h];

        temp[2 * i] = (sum + difference) / sqrt(2);
        temp[2 * i + 1] = (sum - difference) / sqrt(2);
    }

    for (int i = 0; i < length; i++) {
        data[i] = temp[i];
    }

    free(temp);
}

void haar_2d(double* data, int width, int height, int levels) {
    double* row = (double*)malloc(width * sizeof(double));
    double* col = (double*)malloc(height * sizeof(double));

    for (int level = 0; level < levels; level++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                row[x] = data[y * width + x];
            }

            haar_1d(row, width);

            for (int x = 0; x < width; x++) {
                data[y * width + x] = row[x];
            }
        }

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                col[y] = data[y * width + x];
            }

            haar_1d(col, height);

            for (int y = 0; y < height; y++) {
                data[y * width + x] = col[y];
            }
        }

        width /= 2;
        height /= 2;
    }

    free(row);
    free(col);
}

void inverse_haar_2d(double* data, int width, int height, int levels) {
    double* row = (double*)malloc(width * sizeof(double));
        double* col = (double*)malloc(height * sizeof(double));

    for (int level = 0; level < levels; level++) {
        int current_width = width / (1 << (levels - level - 1));
        int current_height = height / (1 << (levels - level - 1));

        for (int x = 0; x < current_width; x++) {
            for (int y = 0; y < current_height; y++) {
                col[y] = data[y * width + x];
            }

            inverse_haar_1d(col, current_height);

            for (int y = 0; y < current_height; y++) {
                data[y * width + x] = col[y];
            }
        }

        for (int y = 0; y < current_height; y++) {
            for (int x = 0; x < current_width; x++) {
                row[x] = data[y * width + x];
            }

            inverse_haar_1d(row, current_width);

            for (int x = 0; x < current_width; x++) {
                data[y * width + x] = row[x];
            }
        }
    }

    free(row);
    free(col);
}

