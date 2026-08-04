# Codex-Context-RW

This document will serve as a manual place where I'm going to input all of my commands and findings in the future. This will not be the prompts that I input, rather just the information or learnings that I want to add in-case I ever come back to the project at a different time. It's both for the sake of context-management, prompt-sharing, and also general learnings that would be useful to refine my prompting. Also, moving forward, I'm going to be deprecating the Project Construction and Company Preparation documents. These will be migrated over to Markdown and/or PDF documents that I'm going to store in Misc.-Directory-2.

## Initial Pull

``llama.cpp/ggml`` -- github.com/ggml-org/llama.cpp and github.com/ggml-org/ggml (backends: ggml-cuda, ggml-metal, ggml-vulkan, ggml-sycl, ggml-cpu, CANN) 

- This is going to be the first set of open-source kernels that I pull in. This is from the official ``llama.cpp`` repository, where I'm going to try and optimize them in any way possible. I'm going to learn what it means to optimize on three specific architectures: Blackwell RTX 6000, A100, and H100/200s. (Note: I bunched together H100 and H200 since they have functionally the same architecture.) Additionally, I'm going to see if I can use the FASRC cluster to test other GPU architectures such as AMD or Intel or Metal/Vulkan or other variations of NVIDIA GPUs.

### ``lamma.cpp/ggml`` Kernel List -- First Sweep

- Completed ``softmax.cu`` and ``scale.cu``. 



