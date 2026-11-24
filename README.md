# Temu 5080

Temu 5080 is an FPGA triangle rasterizer for the AMD Kria KR260. It draws triangles with interpolated vertex colors into a 320 × 240 framebuffer. The included PYNQ notebook uses the rasterizer to render a cube and display the result in Jupyter.

## How rendering works

Python rotates and projects the cube vertices to screen coordinates, orders the triangles by depth, and sends them to the FPGA through AXI DMA. Each triangle consists of nine values in the order `x, y, color` for each of its three vertices. Colors are packed as `0x00RRGGBB`.

The FPGA clips the triangle's bounding box to the framebuffer, uses edge tests to determine which pixels are covered, and interpolates their RGB values using barycentric weights. It writes the resulting pixels to memory over AXI. The notebook reads the framebuffer and displays it with Matplotlib.

## Run the demo

1. On a KR260 running PYNQ, open Jupyter with a Python 3 kernel that has `pynq`, `numpy`, and `matplotlib` available.
2. Place `GPUPYNQ.ipynb`, `softGPU.bit`, and `softGPU.hwh` in the same directory.
3. Open `GPUPYNQ.ipynb` and run the cell that prints `Initializing 3D Pipeline...` for a single cube render. Run the cell that prints `Initializing High-Speed 3D Pipeline...` for the rotating cube. Each cell loads the overlay and allocates its own buffers.

## Repository files

| File | Purpose |
| --- | --- |
| [GPUPYNQ.ipynb](GPUPYNQ.ipynb) | PYNQ host code and rendering demos |
| [soft_gpu.cpp](soft_gpu.cpp) | HLS C++ source for the triangle rasterizer |
| [softGPU.bit](softGPU.bit) | FPGA bitstream for the KR260 |
| [softGPU.hwh](softGPU.hwh) | Hardware metadata for the PYNQ overlay |
