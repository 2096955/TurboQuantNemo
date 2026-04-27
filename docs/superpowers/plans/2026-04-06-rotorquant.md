# RotorQuant KV Compression Implementation Plan

> **ARCHIVED:** Superseded by IsoQuant (renamed from RotorQuant). The Clifford-algebra rotor
> approach was simplified to SO(4) block rotations. See the canonical plan at
> `2026-04-24-isoquant-decode-performance.md`.

**For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement RotorQuant KV compression to speed up TurboQuant via Clifford algebra rotors ($Cl(3,0)$).

**Architecture:**
1. Build `RotorQuantCompressor` in `mlx_turboquant.py` that mirrors `TurboQuantCompressor`.
2. Implement 3D chunking and the Clifford rotor sandwich product (`v' = R v \tilde{R}`) using native MLX operations.
3. Create `RotorQuantKVCache` that inherits from `KVCache` and utilizes the new compressor.
4. Update `server.py` and `--kv-cache-type` CLI arguments to support `"rotorquant"`.

**Tech Stack:** MLX, Python, Geometric Algebra (Clifford).

---

## Chunk 1: Implement RotorQuantCompressor Math

**Files:**
- Modify: `mlx-lm/mlx_lm/models/mlx_turboquant.py`
- Modify: `mlx-lm/tests/test_mlx_turboquant.py` (Create if needed)

- [ ] **Step 1: Write the failing test for `RotorQuantCompressor`**

```python
import mlx.core as mx
from mlx_lm.models.mlx_turboquant import RotorQuantCompressor

def test_rotorquant_compressor():
    # Setup compressor with 3-bit quant
    head_dim = 128
    compressor = RotorQuantCompressor(bit_width=3, head_dim=head_dim)

    # 1. Test compression forward pass
    x = mx.random.normal((2, 10, head_dim))

    # Needs a rotor matrix of shape (head_dim//3, 4) - 4 non-zero components per 3D chunk
    # Just mock a rotation for now
    rotors = mx.ones((head_dim // 3, 4))

    compressed = compressor.compress(x, rotors)
    assert "indices" in compressed
    assert "x_norm" in compressed

    # 2. Test decompressed shapes
    assert compressed["indices"].shape == (2, 10, head_dim)
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement `RotorQuantCompressor` in `mlx_turboquant.py`**

```python
class RotorQuantCompressor:
    """
    RotorQuant Compressor using Clifford $Cl(3,0)$ rotors.
    """
    def __init__(
        self,
        bit_width: int,
        head_dim: int,
        codebook_dir: str = "codebooks",
        seed: int = 42,
    ):
        self.head_dim = head_dim
        self.bits = bit_width
        self.centroids, self.boundaries = load_codebook(head_dim, bit_width, codebook_dir)

        # We need head_dim to be a multiple of 3 for exactly mapping 3D chunks.
        # Pad with zeros if necessary during runtime.
        self.padded_dim = ((head_dim + 2) // 3) * 3
        self.num_chunks = self.padded_dim // 3

    def _sandwich_product(self, x: mx.array, rotors: mx.array) -> mx.array:
        """
        Applies R * x * R_tilde using 4 non-zero components of Cl(3,0) rotors.
        x: [..., num_chunks, 3]
        rotors: [num_chunks, 4]  -> (scalar, bivector_xy, bivector_yz, bivector_zx)
        """
        # Expand rotors to match batch/seq dims of x
        R = mx.expand_dims(rotors, tuple(range(x.ndim - 2)))

        # Rotor components: R0 (scalar), R12, R23, R31 (bivectors)
        R0 = R[..., 0]
        R12 = R[..., 1]
        R23 = R[..., 2]
        R31 = R[..., 3]

        # Vector components
        x1 = x[..., 0]
        x2 = x[..., 1]
        x3 = x[..., 2]

        # Calculate sandwich product (optimized explicit formula for 3D vectors)
        # v' = v + 2 * R_bivector x (v x R_bivector + R_scalar * v)
        # Note: MLX implementation of cross products or direct algebraic expansion goes here
        # For simplicity, we can just do a block-diagonal rotation if precise Clifford expansion is complex,
        # but the spec asks for the sandwich product.

        # Standard 3D rotation from quaternion (rotor) components (w, x, y, z) where w=R0, x=R23, y=R31, z=R12
        # v' = v + 2w(v x u) + 2(u x (v x u)) where u is the bivector part

        u1 = R23
        u2 = R31
        u3 = R12

        # v x u
        cross_x1 = x2 * u3 - x3 * u2
        cross_x2 = x3 * u1 - x1 * u3
        cross_x3 = x1 * u2 - x2 * u1

        # 2w(v x u)
        term1_1 = 2 * R0 * cross_x1
        term1_2 = 2 * R0 * cross_x2
        term1_3 = 2 * R0 * cross_x3

        # u x (v x u)
        cross2_1 = u2 * cross_x3 - u3 * cross_x2
        cross2_2 = u3 * cross_x1 - u1 * cross_x3
        cross2_3 = u1 * cross_x2 - u2 * cross_x1

        term2_1 = 2 * cross2_1
        term2_2 = 2 * cross2_2
        term2_3 = 2 * cross2_3

        out1 = x1 + term1_1 + term2_1
        out2 = x2 + term1_2 + term2_2
        out3 = x3 + term1_3 + term2_3

        return mx.stack([out1, out2, out3], axis=-1)

    def compress(self, x: mx.array, rotors: mx.array) -> dict:
        x_f32 = x.astype(mx.float32)
        x_norm = mx.linalg.norm(x_f32, axis=-1, keepdims=True)
        x_unit = x_f32 / mx.maximum(x_norm, mx.array(1e-8))

        # Pad to multiple of 3
        pad_len = self.padded_dim - self.head_dim
        if pad_len > 0:
            zeros = mx.zeros(x_unit.shape[:-1] + (pad_len,))
            x_unit = mx.concatenate([x_unit, zeros], axis=-1)

        x_chunked = mx.reshape(x_unit, (*x_unit.shape[:-1], self.num_chunks, 3))

        # Apply rotor rotation
        x_rot_chunked = self._sandwich_product(x_chunked, rotors)

        # Flatten back to 1D array per token
        x_rot = mx.reshape(x_rot_chunked, (*x_rot_chunked.shape[:-2], self.padded_dim))
        if pad_len > 0:
            x_rot = x_rot[..., :self.head_dim]

        indices, x_rot_quant = quantize_scalar(x_rot, self.centroids, self.boundaries)

        return {
            "indices": indices,
            "x_rot_quant": x_rot_quant,
            "x_norm": x_norm,
        }
```

- [ ] **Step 4: Run tests and verify they pass**

- [ ] **Step 5: Commit**

## Chunk 2: Add `RotorQuantKVCache` Class

**Files:**
- Modify: `mlx-lm/mlx_lm/models/mlx_turboquant.py`

- [ ] **Step 1: Write `RotorQuantKVCache` extending `KVCache`**

```python
class RotorQuantKVCache(KVCache):
    """
    RotorQuant KV cache. Uses sparse Clifford rotors instead of dense matrices.
    """
    def __init__(self, bit_width: int = 3, head_dim: int = 128, codebook_dir: str = "codebooks"):
        super().__init__()
        self.compressor = RotorQuantCompressor(bit_width, head_dim, codebook_dir)

        # Initialize random rotors [num_chunks, 4]
        # w^2 + x^2 + y^2 + z^2 = 1 for valid rotation
        raw_rotors = mx.random.normal((self.compressor.num_chunks, 4))
        self.rotors = raw_rotors / mx.linalg.norm(raw_rotors, axis=-1, keepdims=True)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        # Compress
        k_comp = self.compressor.compress(keys, self.rotors)
        v_comp = self.compressor.compress(values, self.rotors)

        # Standard cache append logic follows (simplified here)
        # Need to store packed indices and norms
        pass
```

- [ ] **Step 2: Add to cache types in `utils.py`**
Modify `make_prompt_cache` in `mlx-lm/mlx_lm/models/cache.py` to support `kv_cache_type == "rotorquant"`.

- [ ] **Step 3: Commit**
