FROM ubuntu:22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    libstdc++-11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source code
COPY . /build/knapsack

# Download picojson directly (since git submodules don't work without .git history)
RUN mkdir -p /build/knapsack/third_party/picojson && \
    curl -fsSL -o /build/knapsack/third_party/picojson/picojson.h \
      https://raw.githubusercontent.com/kazuho/picojson/master/picojson.h

# Verify picojson is present
RUN test -f /build/knapsack/third_party/picojson/picojson.h || \
    (echo "ERROR: picojson.h not found!" && exit 1)

# Debug: Verify picojson and metal_api.h are present
RUN echo "=== Verifying required headers ===" && \
    test -f /build/knapsack/third_party/picojson/picojson.h && \
    test -f /build/knapsack/kernels/metal/metal_api.h && \
    echo "All required headers found"

# Build the library (static library only)
RUN cd /build/knapsack/knapsack-library && \
    rm -rf build && mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DUSE_METAL=OFF && \
    cmake --build . --target knapsack -j$(nproc) && \
    echo "=== Installing library and header ===" && \
    cp libknapsack.a /usr/local/lib/ && \
    cp ../include/knapsack_c.h /usr/local/include/ && \
    ls -lh /usr/local/lib/libknapsack.a /usr/local/include/knapsack_c.h

# Verify installation
RUN ls -la /usr/local/lib/ && \
    ls -la /usr/local/include/ || true

# Create minimal artifact image
FROM scratch AS artifacts
COPY --from=builder /usr/local/lib/ /lib/
COPY --from=builder /usr/local/include/ /include/
