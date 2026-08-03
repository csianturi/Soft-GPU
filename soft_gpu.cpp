#include <hls_stream.h>
#include <ap_int.h>
#include <math.h>

#define WIDTH 320
#define HEIGHT 240

struct Command {
    int data;
    ap_uint<1> last;
};

// Helper function to extract RGB from a 32-bit hex color (0x00RRGGBB)
void extract_rgb(int hex_color, float &r, float &g, float &b) {
    r = (float)((hex_color >> 16) & 0xFF);
    g = (float)((hex_color >> 8) & 0xFF);
    b = (float)(hex_color & 0xFF);
}

void soft_gpu(hls::stream<Command>& cmd_stream, volatile int* framebuffer) {
    #pragma HLS INTERFACE axis port=cmd_stream
    #pragma HLS INTERFACE m_axi port=framebuffer offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    Command cmd;

    // 1. FETCH STAGE: Now we read 3 coordinates AND 3 colors
    cmd_stream.read(cmd); int x0 = cmd.data;
    cmd_stream.read(cmd); int y0 = cmd.data;
    cmd_stream.read(cmd); int c0 = cmd.data; // Vertex 0 Color
    
    cmd_stream.read(cmd); int x1 = cmd.data;
    cmd_stream.read(cmd); int y1 = cmd.data;
    cmd_stream.read(cmd); int c1 = cmd.data; // Vertex 1 Color
    
    cmd_stream.read(cmd); int x2 = cmd.data;
    cmd_stream.read(cmd); int y2 = cmd.data;
    cmd_stream.read(cmd); int c2 = cmd.data; // Vertex 2 Color

    // 2. SETUP STAGE: Bounding Box and Area Pre-calculation
    int min_x = (x0 < x1) ? ((x0 < x2) ? x0 : x2) : ((x1 < x2) ? x1 : x2);
    int max_x = (x0 > x1) ? ((x0 > x2) ? x0 : x2) : ((x1 > x2) ? x1 : x2);
    int min_y = (y0 < y1) ? ((y0 < y2) ? y0 : y2) : ((y1 < y2) ? y1 : y2);
    int max_y = (y0 > y1) ? ((y0 > y2) ? y0 : y2) : ((y1 > y2) ? y1 : y2);

    if (min_x < 0) min_x = 0; if (min_y < 0) min_y = 0;
    if (max_x >= WIDTH) max_x = WIDTH - 1; if (max_y >= HEIGHT) max_y = HEIGHT - 1;

    // Calculate Triangle Area for Barycentric Interpolation
    float area = (float)((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0));
    if (area == 0.0f) return; // Prevent divide by zero on flat triangles
    float inv_area = 1.0f / area; // Do the division ONCE outside the loop!

    // Extract colors into floats for math
    float r0, g0, b0, r1, g1, b1, r2, g2, b2;
    extract_rgb(c0, r0, g0, b0);
    extract_rgb(c1, r1, g1, b1);
    extract_rgb(c2, r2, g2, b2);

    // 3. RASTERIZATION STAGE
    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            #pragma HLS PIPELINE II=1

            int w0 = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0);
            int w1 = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1);
            int w2 = (x0 - x2) * (y - y2) - (y0 - y2) * (x - x2);

            if ((w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0)) {
                
                // Calculate normalized weights (multiplying by inv_area is faster than dividing)
                // Note: The weights correspond to the opposite vertex
                float alpha = (float)abs(w1) * abs(inv_area);
                float beta  = (float)abs(w2) * abs(inv_area);
                float gamma = (float)abs(w0) * abs(inv_area);

                // Interpolate RGB channels
                int final_r = (int)(alpha * r0 + beta * r1 + gamma * r2);
                int final_g = (int)(alpha * g0 + beta * g1 + gamma * g2);
                int final_b = (int)(alpha * b0 + beta * b1 + gamma * b2);

                // Pack back into 32-bit hex and write to memory
                int final_color = (final_r << 16) | (final_g << 8) | final_b;
                framebuffer[y * WIDTH + x] = final_color;
            }
        }
    }
}