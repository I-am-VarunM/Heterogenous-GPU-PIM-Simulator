# Heterogenous-GPU-PIM-Simulator

> ⚠️ **Project Status**: This project is currently under active development. Features and documentation are being continuously updated.

A novel emulator that combines GPU capabilities with Processing-In-Memory (PIM) architecture, specifically designed for Large Language Model computations. The emulator features a configurable memristor-based crossbar array for efficient vector-matrix multiplications.

## Key Features

- Heterogeneous architecture combining GPU and PIM capabilities
- Configurable memristor crossbar parameters (dimensions and quantity)  
- CUDA-based parallel processing for matrix operations
- Concurrent execution using CUDA streams
- Dedicated GPU module for Softmax and non-constant matrix multiplications
- Comprehensive API for crossbar interaction and customization

## System Architecture

### GPU-PIM Emulator
## System Architecture

| GPU Module | PIM Module |
|------------|------------|
| - Softmax<br>- Matrix Operations | - Crossbar Array<br>- Vector-Matrix Multiplication |

## Getting Started

### Prerequisites
- CUDA Toolkit (version X.X+)
- C++ compiler with C++11 support
