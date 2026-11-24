#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

#define INPUT_HEIGHT 32
#define INPUT_WIDTH  32
#define K            5
#define OUTPUT_CONV1 6
#define OUTPUT_HEIGHT1 28
#define OUTPUT_WIDTH1  28

extern "C" {
void conv1_accel(
    const float* input,     // size 1*32*32
    const float* weights,   // size 6*1*5*5
    const float* bias,      // size 6
    float* output           // size 6*28*28
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


        // Local buffers
        float w_local[OUTPUT_CONV1][K][K];
        #pragma HLS ARRAY_PARTITION variable=w_local complete dim=2
        #pragma HLS ARRAY_PARTITION variable=w_local complete dim=3

        float b_local[OUTPUT_CONV1];
        #pragma HLS ARRAY_PARTITION variable=b_local complete dim=1

        // Line buffer: K rows * full width
        float linebuf[K][INPUT_WIDTH];
        #pragma HLS ARRAY_PARTITION variable=linebuf complete dim=1

        // Load weights + bias from DDR into local memory
        for (int co = 0; co < OUTPUT_CONV1; co++) {
            #pragma HLS PIPELINE
            b_local[co] = bias[co];
        }

        for (int co = 0; co < OUTPUT_CONV1; co++) {
            for (int m = 0; m < K; m++) {
                for (int n = 0; n < K; n++) {
                    #pragma HLS PIPELINE
                    int w_idx = co * (K*K) + m * K + n;
                    w_local[co][m][n] = weights[w_idx];
                }
            }
        }

        // Prime the line buffer with the first K input rows
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < INPUT_WIDTH; j++) {
                #pragma HLS PIPELINE
                linebuf[i][j] = input[i * INPUT_WIDTH + j];
            }
        }

        // Compute outputs, 2 channels at a time
        for (int h = 0; h < OUTPUT_HEIGHT1; h++) {

            // Slide line buffer down by 1 row each output row
            if (h > 0) {
                // shift rows up
                for (int i = 0; i < K-1; i++) {
                    for (int j = 0; j < INPUT_WIDTH; j++) {
                        #pragma HLS PIPELINE
                        linebuf[i][j] = linebuf[i+1][j];
                    }
                }
                // load new bottom row
                int new_row = h + K - 1; // last needed input row for this h
                for (int j = 0; j < INPUT_WIDTH; j++) {
                    #pragma HLS PIPELINE
                    linebuf[K-1][j] = input[new_row * INPUT_WIDTH + j];
                }
            }

            // channels in pairs: (0,1), (2,3), (4,5)
            for (int co = 0; co < OUTPUT_CONV1; co += 2) {
                for (int w = 0; w < OUTPUT_WIDTH1; w++) {
                    #pragma HLS PIPELINE II=1

                    float sum0 = 0.0f;
                    float sum1 = 0.0f;

                    // 5x5 dot product using on-chip linebuf
                    for (int m = 0; m < K; m++) {
                        #pragma HLS UNROLL
                        for (int n = 0; n < K; n++) {
                            // not unrolled to avoid resource blow-up
                            float pix = linebuf[m][w + n];
                            sum0 += w_local[co][m][n]     * pix;
                            sum1 += w_local[co + 1][m][n] * pix;
                        }
                    }

                    int out_base = (OUTPUT_HEIGHT1 * OUTPUT_WIDTH1);
                    int out0 = co       * out_base + h * OUTPUT_WIDTH1 + w;
                    int out1 = (co + 1) * out_base + h * OUTPUT_WIDTH1 + w;

                    output[out0] = sum0 + b_local[co];
                    output[out1] = sum1 + b_local[co + 1];
                }
            }
        }
    }
}
