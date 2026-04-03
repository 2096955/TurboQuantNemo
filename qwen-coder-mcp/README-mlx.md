# Qwen2.5-Coder-32B with MLX

## 1. Convert (download + quantize)

Your original command used `mlx-lm convert --mlx-quantize q4_k_m -o ...`. The installed `mlx_lm` uses different flags. Equivalent command:

```bash
python -m mlx_lm convert \
  --hf-path Qwen/Qwen2.5-Coder-32B-Instruct \
  -q \
  --quant-predicate mixed_4_6 \
  --mlx-path qwen-coder-32b-mlx-q4 \
  --trust-remote-code
```

- `-q` enables quantization.
- `--quant-predicate mixed_4_6` approximates Q4_K_M (mixed 4-bit/6-bit).
- `--mlx-path` is the output path (no `-o` in this CLI).
- `--trust-remote-code` is required for Qwen.

Plain 4-bit (no mixed recipe):

```bash
python -m mlx_lm convert \
  --hf-path Qwen/Qwen2.5-Coder-32B-Instruct \
  -q \
  --q-bits 4 \
  --mlx-path qwen-coder-32b-mlx-q4 \
  --trust-remote-code
```

## 2. Serve

After conversion finishes:

```bash
python -m mlx_lm server --model qwen-coder-32b-mlx-q4 --port 8080
```

Or use the script:

```bash
chmod +x serve-qwen-mlx.sh
./serve-qwen-mlx.sh qwen-coder-32b-mlx-q4 8080
```

API will be at `http://127.0.0.1:8080` (OpenAI-compatible endpoints).

## Check conversion progress

Conversion runs in the background. To watch progress:

```bash
# list terminal output
ls -la ~/.cursor/projects/Users-anthonylui-QwenCoderLocal-qwen-coder-mcp/terminals/

# when done, the directory will exist
ls -la qwen-coder-32b-mlx-q4/
```
