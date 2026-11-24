#include <stdint.h>

#define MATRIXSIZE 16

// Goal: Make memory wider, create parallel computational hardware, pipeline

void matrixMultFast(int32_t A[MATRIXSIZE][MATRIXSIZE], int32_t B[MATRIXSIZE][MATRIXSIZE], int32_t C[MATRIXSIZE][MATRIXSIZE])
{

    #pragma HLS INTERFACE m_axi     port=A  offset=slave bundle=data0
    #pragma HLS INTERFACE m_axi     port=B  offset=slave bundle=data1
    #pragma HLS INTERFACE m_axi     port=C  offset=slave bundle=data2

    #pragma HLS INTERFACE s_axilite register port=A    bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=B    bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=C    bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    int32_t Abuffer[MATRIXSIZE][MATRIXSIZE];
    int32_t Bbuffer[MATRIXSIZE][MATRIXSIZE];   

    #pragma HLS ARRAY_PARTITION variable=Abuffer complete dim=2
    #pragma HLS ARRAY_PARTITION variable=Bbuffer complete dim=1

    for (int i = 0; i < MATRIXSIZE; i++) 
    {
        for (int j = 0; j < MATRIXSIZE; j++) 
        {
            #pragma HLS PIPELINE II=1
            Abuffer[i][j] = A[i][j];
        }
    }

    for (int i = 0; i < MATRIXSIZE; i++) 
    {
        for (int j = 0; j < MATRIXSIZE; j++) 
        {
            #pragma HLS PIPELINE II=1
            Bbuffer[i][j] = B[i][j];
        }
    }

    for(int i = 0; i < MATRIXSIZE; i++)
    {
        for(int j = 0; j < MATRIXSIZE; j++)
        {
            #pragma HLS PIPELINE II=1
            int32_t sum = 0;
            for(int k = 0; k < MATRIXSIZE; k++)
            {
                #pragma HLS UNROLL
                sum += Abuffer[i][k] * Bbuffer[k][j];
            }
            C[i][j] = sum;
        }
    }
}