# knapsack_solver Library

This project implements a solution to the knapsack problem using dynamic programming. The library provides a function `knapsack_solve` that can be used to determine the optimal selection of items to maximize value within a given weight capacity.

## Project Structure

```
knapsack-library
├── src
│   └── knapsack_c.cpp       # Implementation of the knapsack_solve function
├── include
│   └── knapsack_c.h         # Header file declaring the knapsack_solve function
├── CMakeLists.txt           # CMake configuration file for building the library
├── Makefile                 # Makefile for building the library
└── README.md                # Documentation for the project
```

## Building the Library

You can build the knapsack_solver library using either CMake or Makefile. Follow the instructions below based on your preference.

### Using CMake

1. Navigate to the project directory:
   ```bash
   cd /path/to/knapsack-library
   ```

2. Create a build directory and navigate into it:
   ```bash
   mkdir build && cd build
   ```

3. Run CMake to configure the project:
   ```bash
   cmake ..
   ```

4. Build the library:
   ```bash
   make
   ```

### Using Makefile

1. Navigate to the project directory:
   ```bash
   cd /path/to/knapsack-library
   ```

2. Run the command to build the library:
   ```bash
   make
   ```

## Using the Library in Go

After building, the static library (e.g., `libknapsack.a`) will be available for use in other programs written in Go. You can link this library in your Go code using cgo. Here’s a simple example of how to do that:

```go
/*
#cgo LDFLAGS: -L. -lknapsack
#include "knapsack_c.h"
*/
import "C"

// Your Go code to use the knapsack_solve function
```

Make sure to adjust the path to the library as needed.