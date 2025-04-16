# Knapsack

A CUDA-accelerated optimization library for solving complex allocation problems using genetic algorithms and recursive techniques.

## Overview

This project implements an efficient solution to the generalized knapsack problem using CUDA parallelization. The solver utilizes genetic algorithms with mutation and selection to find optimal resource allocation plans in various contexts such as village worker assignment and route planning.

## Features

- GPU-accelerated genetic algorithm implementation
- Recursive solving approach for handling large problem spaces
- Distance-based optimization using haversine formula
- Population-based search with elite selection
- Configurable mutation rates and population sizes
- Debug visualization options

## Requirements

- CUDA Toolkit (11.0+)
- C++17 compatible compiler
- CMake 3.15+
- NVIDIA GPU with compute capability 5.0+

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/knapsack.git
cd knapsack

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make

# Run tests (optional)
make test