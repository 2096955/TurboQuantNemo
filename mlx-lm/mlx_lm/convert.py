# Copyright © 2023-2024 Apple Inc.

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

from .models.switch_layers import SwitchLinear
from .utils import (
    dequantize_model,
    load,
    quantize_model,
    save,
    upload_to_hub,
)


def mixed_quant_predicate_builder(
    recipe: str, model: nn.Module, group_size: int = 64
) -> Callable[[str, nn.Module, dict], Union[bool, dict]]:
    mode = "affine"
    high_bits = 6

    if recipe == "mixed_2_6":
        low_bits = 2
    elif recipe == "mixed_3_4":
        low_bits = 3
        high_bits = 4
    elif recipe == "mixed_3_6":
        low_bits = 3
    elif recipe == "mixed_4_6":
        low_bits = 4
    else:
        raise ValueError(f"Invalid quant recipe {recipe}")

    down_keys = [k for k, _ in model.named_modules() if "down_proj" in k]
    if len(down_keys) == 0:
        raise ValueError("Model does not have expected keys for mixed quant.")

    # Look for the layer index location in the path:
    for layer_location, k in enumerate(down_keys[0].split(".")):
        if k.isdigit():
            break
    num_layers = len(model.layers)

    def mixed_quant_predicate(
        path: str,
        module: nn.Module,
    ) -> Union[bool, dict]:
        """Implements mixed quantization predicates with similar choices to, for example, llama.cpp's Q4_K_M.
        Ref: https://github.com/ggerganov/llama.cpp/blob/917786f43d0f29b7c77a0c56767c0fa4df68b1c5/src/llama.cpp#L5265
        By Alex Barron: https://gist.github.com/barronalex/84addb8078be21969f1690c1454855f3
        """
        index = (
            int(path.split(".")[layer_location])
            if len(path.split(".")) > layer_location
            else 0
        )
        use_more_bits = (
            index < num_layers // 8
            or index >= 7 * num_layers // 8
            or (index - num_layers // 8) % 3 == 2
        )
        if (
            "v_proj" in path or "v_a_proj" in path or "v_b_proj" in path
        ) and use_more_bits:
            return {"group_size": group_size, "bits": high_bits, "mode": mode}
        if "down_proj" in path and use_more_bits:
            return {"group_size": group_size, "bits": high_bits, "mode": mode}
        if "lm_head" in path:
            return {"group_size": group_size, "bits": high_bits, "mode": mode}

        return {"group_size": group_size, "bits": low_bits, "mode": mode}

    return mixed_quant_predicate


QUANT_RECIPES = ["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"]

MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]


def _resolve_quant_defaults(
    mode: str,
    group_size: Optional[int],
    bits: Optional[int],
) -> tuple[int, int]:
    mode_defaults = {
        "affine": (64, 4),
        "mxfp4": (32, 4),
        "nvfp4": (16, 4),
        "mxfp8": (32, 8),
    }
    default_group_size, default_bits = mode_defaults[mode]
    return group_size or default_group_size, bits or default_bits


def _build_mixed_expert_quant_predicate(
    *,
    mixed_expert_bits: int,
    shared_expert_bits: Optional[int] = None,
    default_bits: int,
    default_group_size: int,
    mode: str,
) -> Callable[[str, nn.Module], Union[bool, dict]]:
    if mixed_expert_bits < 1 or mixed_expert_bits > 8:
        raise ValueError("--mixed-expert-bits must be an integer in [1, 8].")
    resolved_shared_bits = (
        shared_expert_bits if shared_expert_bits is not None else default_bits
    )

    def _predicate(path: str, module: nn.Module) -> Union[bool, dict]:
        if ".shared_expert." in path and isinstance(module, nn.Linear):
            return {
                "bits": resolved_shared_bits,
                "group_size": default_group_size,
                "mode": mode,
            }

        if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
            return {"bits": 8, "group_size": default_group_size, "mode": mode}

        # Route lower-bit quantization only to routed expert projections.
        # SwitchLinear is the leaf module at paths like .switch_mlp.fc1,
        # .switch_glu.gate_proj, .switch_mlp.gate_proj etc.
        is_routed_switch = isinstance(module, SwitchLinear) and (
            ".mixer.switch_mlp.fc1" in path
            or ".mixer.switch_mlp.fc2" in path
            or ".switch_glu.gate_proj" in path
            or ".switch_glu.up_proj" in path
            or ".switch_glu.down_proj" in path
            or ".switch_mlp.gate_proj" in path
            or ".switch_mlp.up_proj" in path
            or ".switch_mlp.down_proj" in path
        )
        if is_routed_switch:
            return {
                "bits": mixed_expert_bits,
                "group_size": default_group_size,
                "mode": mode,
            }

        return {"bits": default_bits, "group_size": default_group_size, "mode": mode}

    return _predicate


def _coerce_expert_recipe_arg(recipe: str) -> dict[str, Any]:
    """Resolve --expert-recipe string to a recipe dict (bundled name, path, or 'apex')."""
    if recipe == "apex":
        return {
            "edge_bits": 4,
            "middle_bits": 2,
            "shared_expert": {"bits": 6, "group_size": 64},
            "edge_layer_pct": 0.15,
        }
    path = Path(recipe)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    bundled = Path(__file__).resolve().parent / "recipes" / f"{recipe}.json"
    if bundled.is_file():
        with open(bundled, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(
        f"Unknown --expert-recipe {recipe!r}: not 'apex', not a file path, "
        f"and not mlx_lm/recipes/{recipe}.json"
    )


def _compile_routed_band_schedule(
    bands: List[dict[str, Any]],
    num_layers: int,
    default_group_size: int,
) -> Dict[int, Tuple[int, int]]:
    """Map backbone layer index -> (routed_bits, group_size). Bands are inclusive on start/end."""
    schedule: Dict[int, Tuple[int, int]] = {}
    for b in bands:
        start = int(b["start"])
        end = int(b["end"])
        bits = int(b["routed_bits"])
        gs = int(b.get("routed_group_size", default_group_size))
        if start > end:
            raise ValueError(f"Invalid band start>end: {b}")
        if bits < 1 or bits > 8:
            raise ValueError(f"Invalid routed_bits (expected 1..8): {b}")
        if gs < 1:
            raise ValueError(f"Invalid routed_group_size (expected > 0): {b}")
        for li in range(start, end + 1):
            if li in schedule:
                raise ValueError(f"Overlapping expert bands at layer {li}")
            schedule[li] = (bits, gs)
    missing = [i for i in range(num_layers) if i not in schedule]
    if missing:
        raise ValueError(
            f"Expert recipe bands must cover all layers 0..{num_layers - 1}. "
            f"Missing {len(missing)} layer(s), e.g. {missing[:12]}"
        )
    extra = [i for i in schedule if i < 0 or i >= num_layers]
    if extra:
        raise ValueError(f"Band covers out-of-range layer indices: {extra[:12]}")
    return schedule


def _validate_expert_recipe(
    recipe: dict[str, Any],
    *,
    model_type: str,
    num_layers: int,
) -> None:
    if not isinstance(recipe, dict):
        raise ValueError("--expert-recipe must resolve to a JSON object.")

    schema_version = recipe.get("schema_version")
    if schema_version is not None and int(schema_version) != 1:
        raise ValueError(
            f"Unsupported expert recipe schema_version={schema_version!r}; expected 1."
        )

    recipe_model_family = recipe.get("model_family")
    if recipe_model_family is not None and recipe_model_family != model_type:
        raise ValueError(
            f"Expert recipe model_family={recipe_model_family!r} does not match "
            f"loaded model_type={model_type!r}."
        )

    recipe_layer_count = recipe.get("layer_count")
    if recipe_layer_count is not None and int(recipe_layer_count) != num_layers:
        raise ValueError(
            f"Expert recipe layer_count={recipe_layer_count!r} does not match "
            f"loaded model layers={num_layers}."
        )


def _build_apex_expert_quant_predicate(
    *,
    recipe: dict[str, Any],
    default_bits: int,
    default_group_size: int,
    mode: str,
    num_layers: int,
) -> Callable[[str, nn.Module], Union[bool, dict]]:
    recipe_mode = recipe.get("mode", mode)

    shared_cfg = recipe.get("shared_expert")
    if isinstance(shared_cfg, dict):
        shared_bits = int(shared_cfg["bits"])
        shared_gs = int(shared_cfg.get("group_size", default_group_size))
    else:
        shared_bits = int(recipe.get("shared_bits", 6))
        shared_gs = default_group_size

    routed_schedule: Optional[Dict[int, Tuple[int, int]]] = None
    edge_bits = int(recipe.get("edge_bits", 4))
    middle_bits = int(recipe.get("middle_bits", 2))
    edge_layer_pct = float(recipe.get("edge_layer_pct", 0.15))
    edge_count = int(num_layers * edge_layer_pct)

    if "bands" in recipe:
        routed_schedule = _compile_routed_band_schedule(
            recipe["bands"], num_layers, default_group_size
        )

    def _predicate(path: str, module: nn.Module) -> Union[bool, dict]:
        parts = path.split(".")
        layer_idx = -1
        for p in parts:
            if p.isdigit():
                layer_idx = int(p)
                break

        # SwitchLinear is the leaf module at all routed expert projection paths
        is_routed = isinstance(module, SwitchLinear) and (
            ".mixer.switch_mlp.fc1" in path
            or ".mixer.switch_mlp.fc2" in path
            or ".switch_glu.gate_proj" in path
            or ".switch_glu.up_proj" in path
            or ".switch_glu.down_proj" in path
            or ".switch_mlp.gate_proj" in path
            or ".switch_mlp.up_proj" in path
            or ".switch_mlp.down_proj" in path
        )

        is_shared = ".mixer.shared_experts." in path and isinstance(module, nn.Linear)

        if is_routed:
            if routed_schedule is not None:
                bits, gs = routed_schedule[layer_idx]
            else:
                bits = (
                    edge_bits
                    if layer_idx < edge_count or layer_idx >= (num_layers - edge_count)
                    else middle_bits
                )
                gs = default_group_size
            return {"bits": bits, "group_size": gs, "mode": recipe_mode}

        if is_shared:
            return {"bits": shared_bits, "group_size": shared_gs, "mode": recipe_mode}

        return {
            "bits": default_bits,
            "group_size": default_group_size,
            "mode": recipe_mode,
        }

    return _predicate


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    q_mode: str = "affine",
    dtype: Optional[str] = None,
    upload_repo: str = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
    quant_predicate: Optional[
        Union[Callable[[str, nn.Module, dict], Union[bool, dict]], str]
    ] = None,
    mixed_expert_bits: Optional[int] = None,
    shared_expert_bits: Optional[int] = None,
    expert_recipe: Optional[str] = None,
    trust_remote_code: bool = False,
):
    # Check the save path is empty
    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    if mlx_path.exists():
        raise ValueError(
            f"Cannot save to the path {mlx_path} as it already exists."
            " Please delete the file/directory or specify a new path to save to."
        )

    print("[INFO] Loading")
    model, tokenizer, config = load(
        hf_path,
        revision=revision,
        return_config=True,
        tokenizer_config={"trust_remote_code": trust_remote_code},
        lazy=True,
    )

    if expert_recipe is not None and not quantize:
        raise ValueError("--expert-recipe requires --quantize.")

    resolved_q_group_size, resolved_q_bits = _resolve_quant_defaults(
        q_mode, q_group_size, q_bits
    )

    if expert_recipe is not None:
        if quant_predicate is not None or mixed_expert_bits is not None:
            raise ValueError(
                "Cannot specify --expert-recipe along with --quant-predicate or --mixed-expert-bits"
            )
        recipe_dict = _coerce_expert_recipe_arg(expert_recipe)
        _validate_expert_recipe(
            recipe_dict,
            model_type=config.get("model_type", getattr(model, "model_type", "")),
            num_layers=len(model.layers),
        )
        quant_predicate = _build_apex_expert_quant_predicate(
            recipe=recipe_dict,
            default_bits=resolved_q_bits,
            default_group_size=resolved_q_group_size,
            mode=q_mode,
            num_layers=len(model.layers),
        )

    if mixed_expert_bits is not None:
        if quant_predicate is not None:
            raise ValueError(
                "Cannot specify both --quant-predicate and --mixed-expert-bits."
            )

        quant_predicate = _build_mixed_expert_quant_predicate(
            mixed_expert_bits=mixed_expert_bits,
            shared_expert_bits=shared_expert_bits,
            default_bits=resolved_q_bits,
            default_group_size=resolved_q_group_size,
            mode=q_mode,
        )

    if isinstance(quant_predicate, str):
        if q_mode != "affine":
            raise ValueError("Quant predicates only support 'affine' quantization.")
        quant_predicate = mixed_quant_predicate_builder(
            quant_predicate,
            model,
            resolved_q_group_size,
        )

    if dtype is None:
        dtype = config.get("torch_dtype", None)
    if dtype is None and (text_config := config.get("text_config", None)):
        dtype = text_config.get("dtype", None)
    if dtype in MODEL_CONVERSION_DTYPES:
        print("[INFO] Using dtype:", dtype)
        dtype = getattr(mx, dtype)
        cast_predicate = getattr(model, "cast_predicate", lambda _: True)

        def set_dtype(k, v):
            if cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating):
                return v.astype(dtype)
            else:
                return v

        model.update(tree_map_with_path(set_dtype, model.parameters()))

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    if quantize:
        print("[INFO] Quantizing")
        model, config = quantize_model(
            model,
            config,
            resolved_q_group_size,
            resolved_q_bits,
            mode=q_mode,
            quant_predicate=quant_predicate,
        )

    if dequantize:
        print("[INFO] Dequantizing")
        config.pop("quantization", None)
        config.pop("quantization_config", None)
        model = dequantize_model(model)

    save(
        mlx_path,
        hf_path,
        model,
        tokenizer,
        config,
    )

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo)


def configure_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format"
    )

    parser.add_argument(
        "--hf-path",
        "--model",
        type=str,
        help="Path to the model. This can be a local path or a Hugging Face Hub model identifier.",
    )
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    parser.add_argument(
        "-q", "--quantize", help="Generate a quantized model.", action="store_true"
    )
    parser.add_argument(
        "--q-group-size",
        help="Group size for quantization.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--q-bits",
        help="Bits per weight for quantization.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--q-mode",
        help="The quantization mode.",
        type=str,
        default="affine",
        choices=["affine", "mxfp4", "nvfp4", "mxfp8"],
    )
    parser.add_argument(
        "--quant-predicate",
        help="Mixed-bit quantization recipe.",
        choices=QUANT_RECIPES,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--mixed-expert-bits",
        help="Mixed-precision quantization for routed experts. If set, experts get this bit width (e.g. 2) and everything else gets the default q-bits.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--shared-expert-bits",
        type=int,
        default=None,
        help="Bit-width for shared expert projections (default: same as --q-bits). Use 8 for maximum quality on always-active shared experts.",
    )
    parser.add_argument(
        "--expert-recipe",
        help=(
            "Layer-aware expert quantization: 'apex' (default edge/middle/shared schedule), "
            "path to a JSON recipe, or a bundled name such as 'nemotron_layer_bands_v1' "
            "(see mlx_lm/recipes/)."
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dtype",
        help="Type to save the non-quantized parameters. Defaults to config.json's `torch_dtype` or the current model weights dtype.",
        type=str,
        choices=MODEL_CONVERSION_DTYPES,
        default=None,
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--dequantize",
        help="Dequantize a quantized model.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--trust-remote-code",
        help="Trust remote code when loading tokenizer.",
        action="store_true",
        default=False,
    )
    return parser


def main():
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.convert ...` directly is deprecated."
        " Use `mlx_lm.convert ...` or `python -m mlx_lm convert ...` instead."
    )
    main()
