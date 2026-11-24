#include <stdint.h>

#define INPUT_CONV3     6
#define INPUT_HEIGHT3  14
#define INPUT_WIDTH3   14
#define K3              5

#define OUTPUT_CONV3   16
#define OUTPUT_HEIGHT3 10
#define OUTPUT_WIDTH3  10

extern "C" {
void conv3_accel(
    const float* input,     // [6][14][14]
    const float* weights,   // [16][6][5][5]
    const float* bias,      // [16]
    float* output            // [16][10][10]
) {
    #pragma HLS INTERFACE m_axi port=input   offset=slave bundle=data0
    #pragma HLS INTERFACE m_axi port=weights offset=slave bundle=data1
    #pragma HLS INTERFACE m_axi port=bias    offset=slave bundle=data2
    #pragma HLS INTERFACE m_axi port=output  offset=slave bundle=data3

    #pragma HLS INTERFACE s_axilite port=input   bundle=control
    #pragma HLS INTERFACE s_axilite port=weights bundle=control
    #pragma HLS INTERFACE s_axilite port=bias    bundle=control
    #pragma HLS INTERFACE s_axilite port=output  bundle=control
    #pragma HLS INTERFACE s_axilite port=return  bundle=control


    #pragma HLS INTERFACE s_axilite port=return bundle=control

        // Local buffers
        float w_local[OUTPUT_CONV3][INPUT_CONV3][K3][K3];
        #pragma HLS ARRAY_PARTITION variable=w_local complete dim=3
        #pragma HLS ARRAY_PARTITION variable=w_local complete dim=4

        float b_local[OUTPUT_CONV3];
        #pragma HLS ARRAY_PARTITION variable=b_local complete dim=1

        // Line buffers: [ci][k][width]
        float linebuf[INPUT_CONV3][K3][INPUT_WIDTH3];
        #pragma HLS ARRAY_PARTITION variable=linebuf complete dim=2

        // Load weights + bias
        for (int co = 0; co < OUTPUT_CONV3; co++) {
            #pragma HLS PIPELINE
            b_local[co] = bias[co];
            for (int ci = 0; ci < INPUT_CONV3; ci++) {
                for (int m = 0; m < K3; m++) {
                    for (int n = 0; n < K3; n++) {
                        int idx = (((co * INPUT_CONV3 + ci) * K3 + m) * K3 + n);
                        w_local[co][ci][m][n] = weights[idx];
                    }
                }
            }
        }

        // Prime line buffers
        for (int ci = 0; ci < INPUT_CONV3; ci++) {
            for (int m = 0; m < K3; m++) {
                for (int j = 0; j < INPUT_WIDTH3; j++) {
                    #pragma HLS PIPELINE
                    linebuf[ci][m][j] =
                        input[(ci * INPUT_HEIGHT3 + m) * INPUT_WIDTH3 + j];
                }
            }
        }

        // Convolution
        for (int h = 0; h < OUTPUT_HEIGHT3; h++) {

            // slide line buffers
            if (h > 0) {
                for (int ci = 0; ci < INPUT_CONV3; ci++) {
                    for (int m = 0; m < K3 - 1; m++) {
                        for (int j = 0; j < INPUT_WIDTH3; j++) {
                            #pragma HLS PIPELINE
                            linebuf[ci][m][j] = linebuf[ci][m + 1][j];
                        }
                    }
                    int new_row = h + K3 - 1;
                    for (int j = 0; j < INPUT_WIDTH3; j++) {
                        #pragma HLS PIPELINE
                        linebuf[ci][K3 - 1][j] =
                            input[(ci * INPUT_HEIGHT3 + new_row) * INPUT_WIDTH3 + j];
                    }
                }
            }

            for (int co = 0; co < OUTPUT_CONV3; co++) {
                for (int w = 0; w < OUTPUT_WIDTH3; w++) {
                    #pragma HLS PIPELINE II=1
                    float sum = 0.0f;

                    // accumulate input channels in pairs
                    for (int ci = 0; ci < INPUT_CONV3; ci += 2) {
                        float acc0 = 0.0f;
                        float acc1 = 0.0f;

                        for (int m = 0; m < K3; m++) {
                            #pragma HLS UNROLL
                            for (int n = 0; n < K3; n++) {
                                float pix0 = linebuf[ci][m][w + n];
                                float pix1 = linebuf[ci + 1][m][w + n];
                                acc0 += w_local[co][ci][m][n]     * pix0;
                                acc1 += w_local[co][ci + 1][m][n] * pix1;
                            }
                        }
                        sum += acc0 + acc1;
                    }

                    int out_idx =
                        (co * OUTPUT_HEIGHT3 + h) * OUTPUT_WIDTH3 + w;
                    output[out_idx] = sum + b_local[co];
                }
            }
        }
    }
}
