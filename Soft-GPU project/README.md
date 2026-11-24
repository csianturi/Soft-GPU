# Soft-GPU

Soft-GPU is an EECS 298 project that implements fixed-size matrix multiplication and CNN convolution kernels on the AMD Xilinx Kria KR260 FPGA platform. The project uses C++ high-level synthesis (HLS) for the accelerator kernels and Python notebooks for host-side experiments.

The matrix multiplication demo compares two FPGA designs against NumPy. The baseline pipelines the accumulation loop. The optimized design buffers the inputs on chip, partitions the buffers, and unrolls the dot product to expose parallelism. Both multiply two 16 × 16 signed 32-bit integer matrices.

## Repository contents

| Path | Contents |
| --- | --- |
| [HLS_src/](HLS_src/) | Baseline and optimized matrix multiplication kernels, plus two 5 × 5 convolution kernels for CNN layers. |
| [bit_hwh_files/](bit_hwh_files/) | PYNQ overlay pairs for the KR260. `slowMult` and `fastMult` contain the matrix multiplication designs, and `fasterConv` contains both convolution kernels. |
| [JupyterNotebooks/matMul.ipynb](JupyterNotebooks/matMul.ipynb) | Hardware setup, correctness checks, and timing comparisons against NumPy. |
| [JupyterNotebooks/optimizedImplementation.ipynb](JupyterNotebooks/optimizedImplementation.ipynb) | CNN experiment with image loading, padding, ReLU, and max pooling. It expects an external `images.bin` file and references pooling outputs that the notebook does not define. |
| [EECS298_final_project_report.pdf](EECS298_final_project_report.pdf) | Final project report. |

## Run the matrix multiplication demo

1. Use a KR260 running a PYNQ environment with Jupyter and NumPy.
2. Copy `JupyterNotebooks/` and `bit_hwh_files/` to the board, keeping each `.bit` file next to its matching `.hwh` file.
3. Open `JupyterNotebooks/matMul.ipynb` in Jupyter on the board. In the two `Overlay(...)` calls, replace the original absolute paths with the paths to your copies of `slowMult.bit` and `fastMult.bit`. If the notebook runs from its own directory, use `../bit_hwh_files/slowMult.bit` and `../bit_hwh_files/fastMult.bit`.
4. Run the cells in order. The notebook loads both overlays in sequence, allocates shared buffers, starts each accelerator through its AXI-Lite registers, and compares its output against NumPy.

## Recorded matrix multiplication results

The saved notebook output reports 50 randomized trials per design. Both FPGA designs matched NumPy in every trial.

| Design | Mean NumPy time | Mean FPGA time measured from Python | NumPy time ÷ FPGA time |
| --- | ---: | ---: | ---: |
| Baseline | 48.27 µs | 65.39 µs | 0.74× |
| Optimized | 46.28 µs | 61.48 µs | 0.75× |

The optimized FPGA design was faster than the baseline in this run. NumPy had the lowest measured latency for these 16 × 16 inputs.

## Hardware source

The supplied bitstreams were built with Vivado 2025.1. The HLS entry points are `matrixMult`, `matrixMultFast`, `conv1_accel`, and `conv3_accel`. Each kernel uses AXI memory-mapped interfaces for buffers and AXI-Lite for control. Rebuilding an overlay requires synthesizing and exporting the chosen kernels as IP, then integrating them into a Vivado block design for the KR260. The repository supplies the kernel source and generated overlays. The Vivado project and build scripts are absent.
