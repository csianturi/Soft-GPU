#include <stdint.h>

#define MATRIXSIZE 16

void matrixMult(int32_t A[MATRIXSIZE][MATRIXSIZE], int32_t B[MATRIXSIZE][MATRIXSIZE], int32_t C[MATRIXSIZE][MATRIXSIZE])
{

    #pragma HLS INTERFACE m_axi     port=A  offset=slave bundle=data0
    #pragma HLS INTERFACE m_axi     port=B  offset=slave bundle=data1
    #pragma HLS INTERFACE m_axi     port=C  offset=slave bundle=data2

    #pragma HLS INTERFACE s_axilite port=A    bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=B    bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=C    bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    for(int i = 0; i < MATRIXSIZE; i++)
    {
        for(int j = 0; j < MATRIXSIZE; j++)
        {
            int32_t sum = 0;
            for(int k = 0; k < MATRIXSIZE; k++)
            {
                #pragma HLS PIPELINE II=1
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}