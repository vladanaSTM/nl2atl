# Troubleshooting

## Torch / CUDA import errors

If you see errors like:

```
ImportError: ... libtorch_cuda.so: undefined symbol: ncclMemFree
```

This indicates a mismatch between your installed torch build and CUDA/NCCL runtime. For non-GPU tasks, use the consolidated CLI subcommands; they are lazy-loaded and won’t import torch unless needed.

### Tips

- Ensure your CUDA toolkit and NVIDIA drivers are compatible with the torch build.
- Reinstall torch using the official selector for your CUDA version.

## CLI argument errors

Use the subcommand form:

```bash
nl2atl run-single --model qwen-3b --few_shot
```

Do not pass a script name as a subcommand.
