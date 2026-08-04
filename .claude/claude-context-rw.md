# Claude-Context-RW

This document will serve as a manual place where I'm going to input all of my commands and findings in the future. This will not be the prompts that I input, rather just the information or learnings that I want to add in-case I ever come back to the project at a different time. It's both for the sake of context-management, prompt-sharing, and also general learnings that would be useful to refine my prompting. Also, moving forward, I'm going to be deprecating the Project Construction and Company Preparation documents. These will be migrated over to Markdown and/or PDF documents that I'm going to store in Misc.-Directory-2.

## Initial Pull

``llama.cpp/ggml`` -- github.com/ggml-org/llama.cpp and github.com/ggml-org/ggml (backends: ggml-cuda, ggml-metal, ggml-vulkan, ggml-sycl, ggml-cpu, CANN) 

- This is going to be the first set of open-source kernels that I pull in. This is from the official ``llama.cpp`` repository, where I'm going to try and optimize them in any way possible. I'm going to learn what it means to optimize on three specific architectures: Blackwell RTX 6000, A100, and H100/200s. (Note: I bunched together H100 and H200 since they have functionally the same architecture.) Additionally, I'm going to see if I can use the FASRC cluster to test other GPU architectures such as AMD or Intel or Metal/Vulkan or other variations of NVIDIA GPUs.

### ``lamma.cpp/ggml`` Kernel List -- First Sweep

- Completed ``softmax.cu`` and ``scale.cu``. 

### Kernel Optimization Log (maintained by Claude Code)

One entry per finished kernel: what changed, why it should be faster, and what still needs hardware validation. Baseline for all entries: llama.cpp upstream commit ``2f56fc3`` (2026-08-04).

- ``softmax.cu`` — branch ``perf/softmax-online-fallback``, PR #2 (open).
  - Change: online softmax (Milakov & Gimelshein 2018) for the ``use_shared=false`` fallback path only — rows too long for shared memory, where scratch lives in ``dst`` in global memory. Each thread keeps a running (max, sum) pair and rescales the sum when its max grows, fusing the max pass and the exp+sum pass.
  - Why faster: global traffic drops from 3 reads + 3 writes to 2 reads + 2 writes per element (~33%) on a bandwidth-bound path. Cost: one extra ``expf`` per element (MUFU is not the limiter at these row lengths). Masked ``-inf`` elements skip ``expf`` entirely.
  - Scope/risk: shared-memory hot path byte-for-byte unchanged (``if constexpr`` discard). Sinks, NaN propagation, fully-masked-row NaN, and the #26385 ``__syncthreads`` guard all match upstream. Unvalidated on hardware: run ``test-backend-ops -o SOFT_MAX`` (correctness) and ``perf`` mode. Fallback triggers when ``(ncols padded to 32) + 32`` floats exceed smpbo (~25K cols on consumer, ~40-56K on A100/H100).

- ``scale.cu`` — branch ``perf/scale-float4``, PR #3 (open).
  - Change: added ``scale_f32_v4`` processing 4 elements/thread via ``float4`` (16-byte accesses) with an in-kernel scalar tail loop; host dispatches on a 16-byte alignment check of both pointers, otherwise falls back to the unchanged scalar kernel.
  - Why faster: pure streaming kernel; 4x fewer memory instructions per byte and wider transactions improve achieved bandwidth, mainly at low occupancy / small-to-mid tensors. Bitwise-identical results (same FFMA per element).
  - Scope/risk: no ``__restrict__`` on purpose (``ggml_scale_inplace`` runs with ``dst == x``). PDL lc/sync structure preserved. Unvalidated: ``test-backend-ops -o SCALE``.

- ``clamp.cu`` — branch ``perf/clamp-vec4``, PR #4 (open).
  - Change: added ``op_clamp_kernel_v4`` processing 4 elements/thread through an ``alignas(4*sizeof(T))`` wrapper struct — one 16-byte access for f32, one 8-byte access for f16 — with a scalar tail for ``k % 4``. Host dispatches on pointer alignment; scalar kernel unchanged as fallback.
  - Why faster: 4x fewer memory instructions for both dtypes; f16 especially benefits since 2-byte scalar accesses are the least efficient width. Bitwise-identical results (same per-element float-convert + fminf/fmaxf).
  - Scope/risk: no ``__restrict__`` (in-place clamp). ``k`` stays ``int`` (upstream contract). Unvalidated: ``test-backend-ops -o CLAMP``.
  - Workflow note: the first attempt at this commit swept in staged context-doc edits (docs were git-added before the commit); fixed with ``git reset --soft`` + recommit + force-push. Lesson: check ``git status`` for staged files before committing on perf branches.

- ``cumsum.cu`` — branch ``perf/cumsum-fallback-vec4``, PR #5 (open).
  - Change: two edits to ``cumsum_kernel`` (the non-CUB fallback scan; CUB paths untouched). (1) Removed the ``s_vals`` shared array — each thread only ever read back its own slot, so the warp-scanned value now stays in a register; smem per block drops ~1 KiB and a store+load per tile disappears. (2) Vectorized the 4-consecutive-element register-blocking loads/stores with the same ``alignas`` struct idiom as clamp (16B for f32), guarded per row on alignment with the scalar path kept for tails/unaligned rows.
  - Why faster: less shared-memory traffic and capacity per block, 4x fewer memory instructions on the main path. Sequential-add order unchanged → bitwise-identical results.
  - Scope/risk: launch-side ``shmem_size`` updated in lockstep with the kernel's smem layout (they must move together). Verified ``warp_prefix_inclusive_sum`` is pure shuffle before removing the array. Unvalidated: ``test-backend-ops -o CUMSUM``.
  - Reading note: the ``s_vals`` find is a good pattern to hunt for — shared memory that is written and read only by the same thread is always a register in disguise.

- ``sum.cu`` — branch ``perf/sum-fallback-reduction``, PR #6 (open).
  - Change: replaced the non-CUB fallback (HIP/MUSA builds only; CUB path untouched) that summed the whole tensor via ``sum_rows`` with ``nrows == 1`` — i.e. ONE thread block for the entire reduction — with a two-pass tree reduction: up to 1024 grid-striding blocks write partial sums (pool buffer), then one block reduces the partials. Tensors <= 256 elements take a direct single-block path.
  - Why faster: ~1000x parallelism deficit removed for large tensors on AMD builds. Deterministic by construction (fixed grid geometry), which is why it is two passes and not a one-pass ``atomicAdd``.
  - Scope/risk: results differ from the old fallback in rounding only (different association order). Only compiles/runs on non-CUB builds — needs a HIP build to validate (``test-backend-ops -o SUM``); relevant to the planned FASRC AMD experiments.

- ``rope.cu`` — branch ``perf/rope-launch-geometry``, PR #7 (open). Round 2 begins.
  - Change: host-only launch-geometry fix across all four RoPE launchers (norm/neox/multi/vision). Blocks were a fixed (1, 256) with one thread per rotation pair, but a row has only ``ne00/2`` pairs — at head_dim 128, 75% of every block's lanes failed the bounds check instantly yet stayed resident (a block's thread/register allocation is held for its lifetime). Now ``blockDim.y`` = pair count rounded up to a warp, capped at 256; kernels untouched (they read ``blockDim.y`` dynamically).
  - Why faster: 2-4x more *active* warps per SM on a memory-bound kernel — active warps are what hide DRAM latency. RoPE runs on every layer's Q and K even with flash attention on, so this is the first change on the genuinely hot inference path.
  - Scope/risk: bitwise identical (only never-active threads removed; >256-pair rows reproduce old geometry exactly). Unvalidated: ``test-backend-ops -o ROPE``, ``llama-bench -p 2048``.
  - Future work noted in PR: ``powf(theta_scale, i0/2)`` → ``exp2f`` with host-precomputed log2 (changes rounding; needs its own hardware-validated PR).
  - Reading note: third optimization genre so far — (1) algorithmic (softmax online rescale), (2) access width (float4/alignas), (3) launch geometry (thread blocks sized to actual work). Occupancy claimed by dead lanes is invisible in the code; you find it by comparing the bounds check against the block shape.

- ``mean.cu`` → actually ``reduce_rows.cuh`` — branch ``perf/reduce-rows-vec4``, PR #8 (open).
  - Change: ``mean.cu`` itself is dispatch + upstream-tuned heuristics (PR #15132), so the real target was the shared ``reduce_rows_f32`` kernel serving both MEAN and SUM_ROWS. Vectorized its load loop: 4 columns per ``float4`` load when ``ncols % 4 == 0`` and base is 16B-aligned (two loads/iteration preserves the 8-deep unroll); scalar loop kept verbatim as fallback.
  - Why faster: 4x fewer load instructions; biggest effect in the low-nrows 512-thread regime where the kernel is latency-bound. ``ncols % 4 == 0`` guarantees every row base stays aligned, so the check is once per kernel.
  - Scope/risk: NOT bitwise-identical on the vec path (accumulator lanes regroup → rounding differs; same reassociation reasoning as PR #6). Unvalidated: ``test-backend-ops -o MEAN`` and ``-o SUM_ROWS``.
  - Reading note: when a kernel file turns out to be a dispatch shell, follow the include to where the loads actually happen — the "one file = one kernel" mapping is a convention, not a law.

- ``sumrows.cu`` — branch ``perf/sumrows-dedupe-blocksize``, PR #9 (open).
  - Change: (1) capped the low-nrows branch's 512-thread blocks at ``ncols`` rounded up to a warp — for short rows, hundreds of threads were skipping the load loop and feeding zeros into ``block_reduce``, paying sync cost for nothing; rows >= 512 columns keep the old geometry exactly. (2) Deduplicated: ``ggml_cuda_op_sum_rows`` re-implemented ``sum_rows_f32_cuda`` line for line; it now calls the helper, keeping the PR #15132 heuristic in one place. Net -9 lines.
  - Why faster: fewer reduction participants and scheduler slots for small-row shapes. Bitwise identical up to the sign of zero (removed threads contributed exact 0.0f).
  - Scope/risk: kernel untouched (that was PR #8); public signature unchanged; no other callers. Unvalidated: ``test-backend-ops -o SUM_ROWS``.
  - Reading note: same launch-geometry genre as rope (PR #7) but a different mechanism — rope's excess lanes were idle-but-resident (occupancy theft), these actively participated with zeros (sync/scheduling waste). Same smell, two diseases.

- ``unary.cu`` — branch ``perf/unary-vec4``, PR #10 (open). Merging this = Bronze x2.
  - Change: vectorized ``unary_op_kernel`` — the ONE template behind ~26 elementwise ops (relu/gelu/silu/exp/...) plus fused relu_sqr — with the ``alignas`` vec4 idiom (16B f32 / 8B f16 accesses, in-kernel scalar tail, alignment-dispatched, scalar fallback kept). Highest leverage-per-line of the vectorization PRs: one kernel, 26+ ops.
  - Also fixed two launch-math slips found while reading: silu_back's ceil-divide mixed two different block-size constants (benign only because both are 256 today — a retuning landmine), and xielu's ``(k + BS)/BS`` launched one empty extra block whenever k divides evenly.
  - Scope/risk: bitwise identical (per-element math untouched); no ``__restrict__`` (in-place ops); PDL preserved. Gated/GLU kernels left scalar — their per-row offset indexing needs a row-alignment argument, future PR. Unvalidated: ``test-backend-ops -o UNARY / SILU_BACK / XIELU``.
  - Reading note: launch-config arithmetic is copy-pasted boilerplate nobody reads — which is exactly why two of the file's three hand-written ceil-divides had slips. Boilerplate that must be repeated is boilerplate that will eventually be repeated wrong.

- ``diagmask.cu`` — branch ``perf/diagmask-block-cap``, PR #11 (open).
  - Change: block size raised from a fixed one-warp (1, 32) to 256, capped at the row length rounded to a warp. Two-file diff (define in ``.cuh`` + launcher); kernel untouched.
  - Why faster: the per-SM resident-*block* limit (16-32 by arch) binds before the thread budget with one-warp blocks — 512-1024 resident threads of a 1536-2048 budget, a 25-50% occupancy ceiling from block-slot exhaustion on a memory-bound elementwise kernel.
  - Scope/risk: pure elementwise, zero cross-thread interaction → bitwise identical under ANY launch shape (strongest correctness class for geometry changes). Unvalidated: ``test-backend-ops -o DIAG_MASK_INF``. Op is only hot on non-FA attention paths.
  - Reading note: went in expecting the rope disease (blocks too big) and found the mirror image (blocks too small). Launch geometry fails in both directions: too-big blocks waste allocation on dead lanes; too-small blocks starve the SM via the block-slot limit. Always check BOTH bounds. Also: verify the actual #define before claiming a finding — the hypothesis formed from the launcher's shape was wrong until the constant was read.
  - Badge-math correction (from this session): Pair Extraordinaire tiers 1/10/24/48 (co-authored commits in merged PRs); Pull Shark tiers 2/16/128/1024 (merged PRs authored) — the user earns BOTH per merged PR here. "16" belongs to Pull Shark, not Pair Extraordinaire. GitHub publishes no official numbers; the profile progress bar is ground truth.