# optimized/

Optimized versions of the ggml-cuda kernels. Only files that differ from upstream live here, under the same filename as their original in `../ggml-cuda/`.

- `../ggml-cuda/` is the pristine vendored baseline: llama.cpp `ggml/src/ggml-cuda` at upstream commit `2f56fc3`. It is never edited.
- `optimized/` is a drop-in overlay on top of that baseline.

## Building

Copy the baseline into a llama.cpp checkout at `2f56fc3`, then overlay this directory:

```bash
cp -r Pangu/ggml-cuda/. llama.cpp/ggml/src/ggml-cuda/
cp -r Pangu/optimized/. llama.cpp/ggml/src/ggml-cuda/
rm llama.cpp/ggml/src/ggml-cuda/README.md
```

Skip the second `cp` to build the baseline for an A/B comparison.

## Seeing what changed

```bash
diff -u ggml-cuda/cpy.cu optimized/cpy.cu   # one kernel
diff -ru ggml-cuda optimized | less         # everything ("Only in" lines are untouched files)
```

## Optimizing a new kernel

1. Branch from main: `perf/<kernel>-<what>`.
2. If the file is not here yet, copy it from `../ggml-cuda/` and edit the copy.
3. One PR per kernel, validated on hardware before merge (`test-backend-ops -o <OP>`, then `test-backend-ops perf -o <OP>` with repeats).
