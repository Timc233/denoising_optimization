
#include "basic_util.h"
#include "set_data_type.h"

// Assume that the output array is already allocated
// and the length of the output array is the same or less than the input array
void copy_array(data_t* input, data_t* output, int output_len){
    for (int i = 0; i < output_len; i++) {
        output[i] = input[i];
    }
}

int inplace_division(int n){
    int temp = n;
    return temp / 2;
}