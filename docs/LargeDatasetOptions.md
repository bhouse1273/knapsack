# Large Dataset Ingestion Options

You don’t need one JSON document per row. The V2 API already expects **columnar arrays** (Structure-of-Arrays) so you can pass all 500 K rows in a single config—each attribute is just an array of length `n`. Sending 500 K × 10 (~5 M numbers) as JSON is large (≈100–200 MB), but the library can handle it; the limitation is really about transport and parsing costs on your side.

### Options for large datasets

- **Single SoA JSON (current API)**  
  Create one config where each attribute array has 500 K values. This keeps records aligned by index. It’s the simplest path and already supported, but you pay the JSON parsing cost.

- **Binary or Arrow ingestion (roadmap)**  
  Internally we can load Arrow buffers, Parquet, or raw binaries (float/int arrays) as long as they align with the same SoA layout. If you prefer this, you can add a `source` block in the config that points to files (CSV/Parquet/Arrow) and we’ll extend `BuildHostSoA` to read them directly without expanding JSON.

- **Streaming via C++/CGO**  
  If you’re integrating directly, skip JSON entirely:
  1. Load your data into native arrays (e.g., `std::vector<float>` or Go slices converted to C arrays).
  2. Build the `HostSoA` manually (see Data.cpp).
  3. Call `BeamSearch` or `Evaluate*` functions directly.

- **Chunking for transport, single SoA in-memory**  
  If transport is the bottleneck, you can send compressed chunks (e.g., gzipped JSON lines or Arrow flight). Once on the server, concatenate them into one SoA before calling knapsack; the solver wants a single contiguous block.

### Recommended process for your case (500 K rows)

1. **Server-side loader**: Convert the raw dataset (CSV, Parquet, etc.) into SoA arrays in memory (floats/ints). Use direct file loaders in `BuildHostSoA` to avoid the JSON overhead.
2. **Knapsack config**: Either (a) continue to produce a single JSON with `items.attributes.<field>` arrays, or (b) create a tiny config pointing to the preloaded data (e.g., `"data_source": "file://…/data.arrow"` once we add that hook).
3. **Execution**: Use the existing v2 API (`knapsack_solver --config config.json` or the C++/Go bindings). You only need *one* config run; the SoA arrays already contain all rows.

### Takeaways

- You do **not** need `n` JSON documents; the current API is columnar.
- For better performance, load data in binary format or build the SoA in memory instead of serializing huge JSON arrays.
- If you want file-based ingestion (CSV/Parquet/Arrow), we can add a `data_source` block to the config that tells the solver to read from disk directly—no JSON swelling.

Let me know if you’d like me to prototype the file-backed loader or provide a reference implementation for streaming large datasets into `HostSoA`.