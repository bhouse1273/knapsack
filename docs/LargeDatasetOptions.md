# Large Dataset Ingestion Options

You don’t need one JSON document per row. The V2 API already expects **columnar arrays** (Structure-of-Arrays) so you can pass all 500 K rows in a single config—each attribute is just an array of length `n`. Sending 500 K × 10 (~5 M numbers) as JSON is large (≈100–200 MB), but the library can handle it; the limitation is really about transport and parsing costs on your side.

### Options for large datasets

- **Single SoA JSON (current API)**  
  Create one config where each attribute array has 500 K values. This keeps records aligned by index. It’s the simplest path and already supported, but you pay the JSON parsing cost.

- **Binary / Arrow / Parquet ingestion (built-in)**  
  Use an external `source` spec for any attribute (see `docs/V2_JSON_SCHEMA.md`). The solver can now read raw binary doubles, CSV, Arrow IPC/Feather, or Parquet files directly into the SoA without blowing up the JSON payload.

- **Streaming via C++/CGO**  
  If you’re integrating directly, skip JSON entirely:
  1. Load your data into native arrays (e.g., `std::vector<float>` or Go slices converted to C arrays).
  2. Build the `HostSoA` manually (see Data.cpp).
  3. Call `BeamSearch` or `Evaluate*` functions directly.

- **Chunking for transport, single SoA in-memory**  
  If transport is the bottleneck, you can send compressed chunks (e.g., gzipped JSON lines or Arrow flight). Once on the server, concatenate them into one SoA before calling knapsack; the solver wants a single contiguous block.

### Recommended process for your case (500 K rows)

1. **Server-side loader**: Convert the raw dataset (CSV, Parquet, etc.) into SoA arrays in memory (floats/ints). Use direct file loaders in `BuildHostSoA` to avoid the JSON overhead.
2. **Knapsack config**: Either (a) continue to produce a single JSON with `items.attributes.<field>` arrays, or (b) create a tiny config pointing to the preloaded data via the existing `source` block (e.g., an Arrow or Parquet file on disk).
3. **Execution**: Use the existing v2 API (`knapsack_solver --config config.json` or the C++/Go bindings). You only need *one* config run; the SoA arrays already contain all rows.

### Takeaways

- You do **not** need `n` JSON documents; the current API is columnar.
- For better performance, load data in binary format or build the SoA in memory instead of serializing huge JSON arrays.
- File-based ingestion (CSV/Parquet/Arrow/binary) already works through the `source` block, so you can keep the JSON tiny and let the solver read the heavy payload straight from disk.

### External fixture path for tests

If you keep large reference datasets outside the repo, point the test suite at them with the `KNAPSACK_TESTDATA_ROOT` environment variable (or the matching CMake cache entry):

```bash
# Configure once so CTest always exports the env var
cmake -S . -B build -DKNAPSACK_TESTDATA_ROOT=/Volumes/mtheoryssd/2025-M-Theory/KnapsackTestData

# Or export temporarily when running tests/tools
export KNAPSACK_TESTDATA_ROOT=/Volumes/mtheoryssd/2025-M-Theory/KnapsackTestData
ctest --output-on-failure -R data_ingest
```

When set, the `data_ingest` Catch2 suite automatically converts real fixtures such as `samples/small_mkp.txt` into Arrow/Parquet files to verify ingestion of multi-column datasets without copying those fixtures into the repo.

### Converting OR-Library fixtures into Arrow/Parquet

For larger multi-constraint coverage you can export the OR-Library MKP instances straight into Arrow/Parquet (plus a ready-to-run config) with the new tool:

```bash
# Configure to ensure convert_orlibrary_to_arrow is built
cmake -S . -B build -DKNAPSACK_TESTDATA_ROOT=/Volumes/mtheoryssd/2025-M-Theory/KnapsackTestData
cmake --build build --target convert_orlibrary_to_arrow

# Example: instance 3 (n=20, m=10) and instance 6 (n=50, m=5)
for idx in 3 6; do
  KNAPSACK_TESTDATA_ROOT=/Volumes/mtheoryssd/2025-M-Theory/KnapsackTestData \
    ./build/convert_orlibrary_to_arrow \
      --dataset or-library/mknap1.txt \
      --instance ${idx} \
      --out data/benchmarks/orlib_mknap1_i${idx};
done
```

Each invocation writes three files next to your chosen `--out` prefix:

- `*.arrow`: Arrow IPC file with `value` plus `weight_0…weight_{m-1}` columns.
- `*.parquet`: Identical schema encoded as Parquet for streaming benchmarks.
- `*.json`: V2 config that references the Arrow file via `items.sources` and wires every constraint limit from the raw text.

Because the tool reads directly from `KNAPSACK_TESTDATA_ROOT`, the inputs remain outside the repo while tests/benchmarks can rely on stable Arrow/Parquet artifacts generated on-demand.

Let me know if you’d like me to prototype the file-backed loader or provide a reference implementation for streaming large datasets into `HostSoA`.