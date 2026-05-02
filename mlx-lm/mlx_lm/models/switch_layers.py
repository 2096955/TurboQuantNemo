# Copyright © 2023-2024 Apple Inc.

import math
from typing import TYPE_CHECKING, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu

if TYPE_CHECKING:
    from ..expert_offload import ExpertOffloadManager


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Freeze this model's parameters
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class OffloadSwitchGLU(nn.Module):
    """MoE SwitchGLU with routed experts loaded via ExpertOffloadManager."""

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = OffloadSwitchLinear(
            input_dims,
            hidden_dims,
            num_experts,
            bias=bias,
            proj_name="gate",
        )
        self.up_proj = OffloadSwitchLinear(
            input_dims,
            hidden_dims,
            num_experts,
            bias=bias,
            proj_name="up",
        )
        self.down_proj = OffloadSwitchLinear(
            hidden_dims,
            input_dims,
            num_experts,
            bias=bias,
            proj_name="down",
        )
        self.activation = activation

    def set_expert_manager(
        self, manager: "ExpertOffloadManager", layer_idx: int
    ) -> None:
        self.gate_proj.manager = manager
        self.gate_proj.layer_idx = layer_idx
        self.up_proj.manager = manager
        self.up_proj.layer_idx = layer_idx
        self.down_proj.manager = manager
        self.down_proj.layer_idx = layer_idx

    def __call__(self, x, indices) -> mx.array:
        mgr = self.gate_proj.manager
        if mgr is None:
            raise RuntimeError("OffloadSwitchGLU: ExpertOffloadManager not attached")

        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        gate_w, up_w, down_w, remapped = mgr.prepare_gather_triple(
            self.gate_proj.layer_idx, idx, mode="gather"
        )

        x_up = mx.gather_mm(
            x,
            up_w.swapaxes(-1, -2),
            rhs_indices=remapped,
            sorted_indices=do_sort,
        )
        if "bias" in self.up_proj:
            x_up = x_up + mx.expand_dims(self.up_proj["bias"][idx], -2)

        x_gate = mx.gather_mm(
            x,
            gate_w.swapaxes(-1, -2),
            rhs_indices=remapped,
            sorted_indices=do_sort,
        )
        if "bias" in self.gate_proj:
            x_gate = x_gate + mx.expand_dims(self.gate_proj["bias"][idx], -2)

        x = mx.gather_mm(
            self.activation(x_up, x_gate),
            down_w.swapaxes(-1, -2),
            rhs_indices=remapped,
            sorted_indices=do_sort,
        )
        if "bias" in self.down_proj:
            x = x + mx.expand_dims(self.down_proj["bias"][idx], -2)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        out = x.squeeze(-2)
        mx.eval(out)
        return out


class OffloadSwitchLinear(nn.Module):
    """Routed expert linear backed by ExpertOffloadManager (compact gather_mm)."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        *,
        proj_name: str = "fc1",
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self._num_experts = num_experts
        self.proj_name = proj_name
        self.layer_idx = -1
        self.manager: Optional["ExpertOffloadManager"] = None
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def num_experts(self):
        return self._num_experts

    def __call__(self, x, indices, sorted_indices=False):
        mgr = self.manager
        if mgr is None:
            raise RuntimeError(
                "OffloadSwitchLinear: ExpertOffloadManager not attached (set_expert_manager)"
            )
        compact_w, remapped = mgr.prepare_gather(
            self.layer_idx, self.proj_name, indices, mode="gather"
        )
        out = mx.gather_mm(
            x,
            compact_w.swapaxes(-1, -2),
            rhs_indices=remapped,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            out = out + mx.expand_dims(self["bias"][indices], -2)
        return out


class OffloadSwitchMLP(nn.Module):
    """MoE MLP with routed experts loaded via ExpertOffloadManager."""

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()
        self.fc1 = OffloadSwitchLinear(
            input_dims,
            hidden_dims,
            num_experts,
            bias=bias,
            proj_name="fc1",
        )
        self.fc2 = OffloadSwitchLinear(
            hidden_dims,
            input_dims,
            num_experts,
            bias=bias,
            proj_name="fc2",
        )
        self.activation = activation

    def set_expert_manager(
        self, manager: "ExpertOffloadManager", layer_idx: int
    ) -> None:
        self.fc1.manager = manager
        self.fc1.layer_idx = layer_idx
        self.fc2.manager = manager
        self.fc2.layer_idx = layer_idx

    def __call__(self, x, indices) -> mx.array:
        mgr = self.fc1.manager
        if mgr is None:
            raise RuntimeError("OffloadSwitchMLP: ExpertOffloadManager not attached")
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        compact_fc1, compact_fc2, remapped = mgr.prepare_gather_pair(
            self.fc1.layer_idx, idx, mode="gather"
        )

        x = mx.gather_mm(
            x,
            compact_fc1.swapaxes(-1, -2),
            rhs_indices=remapped,
            sorted_indices=do_sort,
        )
        if "bias" in self.fc1:
            x = x + mx.expand_dims(self.fc1["bias"][idx], -2)
        x = self.activation(x)
        x = mx.gather_mm(
            x,
            compact_fc2.swapaxes(-1, -2),
            rhs_indices=remapped,
            sorted_indices=do_sort,
        )
        if "bias" in self.fc2:
            x = x + mx.expand_dims(self.fc2["bias"][idx], -2)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        out = x.squeeze(-2)
        mx.eval(out)
        return out


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class OffloadQuantizedSwitchLinear(nn.Module):
    """Routed quantized expert linear backed by ExpertOffloadManager."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = False,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        *,
        proj_name: str = "fc1",
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self._num_experts = num_experts
        self.proj_name = proj_name
        self.layer_idx = -1
        self.manager: Optional["ExpertOffloadManager"] = None
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def num_experts(self):
        return self._num_experts

    def __call__(self, x, indices, sorted_indices=False):
        mgr = self.manager
        if mgr is None:
            raise RuntimeError(
                "OffloadQuantizedSwitchLinear: ExpertOffloadManager not attached"
            )
        compact_w, compact_s, compact_b, remapped = mgr.prepare_gather_quantized(
            self.layer_idx,
            self.proj_name,
            indices,
            mode="gather",
        )
        out = mx.gather_qmm(
            x,
            compact_w,
            compact_s,
            compact_b,
            rhs_indices=remapped,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            out = out + mx.expand_dims(self["bias"][indices], -2)
        return out


class OffloadQuantizedSwitchMLP(nn.Module):
    """MoE MLP with quantized routed experts loaded via ExpertOffloadManager."""

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()
        self.fc1 = OffloadQuantizedSwitchLinear(
            input_dims,
            hidden_dims,
            num_experts,
            bias=bias,
            group_size=group_size,
            bits=bits,
            mode=mode,
            proj_name="fc1",
        )
        self.fc2 = OffloadQuantizedSwitchLinear(
            hidden_dims,
            input_dims,
            num_experts,
            bias=bias,
            group_size=group_size,
            bits=bits,
            mode=mode,
            proj_name="fc2",
        )
        self.activation = activation

    def set_expert_manager(
        self, manager: "ExpertOffloadManager", layer_idx: int
    ) -> None:
        self.fc1.manager = manager
        self.fc1.layer_idx = layer_idx
        self.fc2.manager = manager
        self.fc2.layer_idx = layer_idx

    def __call__(self, x, indices) -> mx.array:
        mgr = self.fc1.manager
        if mgr is None:
            raise RuntimeError(
                "OffloadQuantizedSwitchMLP: ExpertOffloadManager not attached"
            )

        if mgr.predictor is not None and hasattr(mgr.predictor, "compute_proxy_alpha"):
            alpha = mgr.predictor.compute_proxy_alpha(self.fc1.layer_idx, x)
            predicted = mgr.predictor.predict_experts(
                self.fc1.layer_idx, alpha, top_k=getattr(mgr, "prefetch_top_k", 2)
            )
            mgr.prefetch(self.fc1.layer_idx, predicted)
            self._last_alpha = alpha

        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        fc1_tensors, fc2_tensors, remapped = mgr.prepare_gather_pair_quantized(
            self.fc1.layer_idx, idx, mode="gather"
        )

        fc1_w, fc1_s, fc1_b = fc1_tensors
        x = mx.gather_qmm(
            x,
            fc1_w,
            fc1_s,
            fc1_b,
            rhs_indices=remapped,
            transpose=True,
            group_size=self.fc1.group_size,
            bits=self.fc1.bits,
            mode=self.fc1.mode,
            sorted_indices=do_sort,
        )
        if "bias" in self.fc1:
            x = x + mx.expand_dims(self.fc1["bias"][idx], -2)

        x = self.activation(x)

        fc2_w, fc2_s, fc2_b = fc2_tensors
        x = mx.gather_qmm(
            x,
            fc2_w,
            fc2_s,
            fc2_b,
            rhs_indices=remapped,
            transpose=True,
            group_size=self.fc2.group_size,
            bits=self.fc2.bits,
            mode=self.fc2.mode,
            sorted_indices=do_sort,
        )
        if "bias" in self.fc2:
            x = x + mx.expand_dims(self.fc2["bias"][idx], -2)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        out = x.squeeze(-2)
        mx.eval(out)

        if mgr.predictor is not None and hasattr(self, "_last_alpha"):
            mgr.predictor.record_activation(self.fc1.layer_idx, self._last_alpha, idx)

        return out


class OffloadQuantizedSwitchGLU(nn.Module):
    """Quantized MoE SwitchGLU with routed experts loaded via ExpertOffloadManager."""

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()
        self.gate_proj = OffloadQuantizedSwitchLinear(
            input_dims,
            hidden_dims,
            num_experts,
            bias=bias,
            group_size=group_size,
            bits=bits,
            mode=mode,
            proj_name="gate",
        )
        self.up_proj = OffloadQuantizedSwitchLinear(
            input_dims,
            hidden_dims,
            num_experts,
            bias=bias,
            group_size=group_size,
            bits=bits,
            mode=mode,
            proj_name="up",
        )
        self.down_proj = OffloadQuantizedSwitchLinear(
            hidden_dims,
            input_dims,
            num_experts,
            bias=bias,
            group_size=group_size,
            bits=bits,
            mode=mode,
            proj_name="down",
        )
        self.activation = activation

    def set_expert_manager(
        self, manager: "ExpertOffloadManager", layer_idx: int
    ) -> None:
        self.gate_proj.manager = manager
        self.gate_proj.layer_idx = layer_idx
        self.up_proj.manager = manager
        self.up_proj.layer_idx = layer_idx
        self.down_proj.manager = manager
        self.down_proj.layer_idx = layer_idx

    def __call__(self, x, indices) -> mx.array:
        mgr = self.gate_proj.manager
        if mgr is None:
            raise RuntimeError(
                "OffloadQuantizedSwitchGLU: ExpertOffloadManager not attached"
            )

        if mgr.predictor is not None and hasattr(mgr.predictor, "compute_proxy_alpha"):
            alpha = mgr.predictor.compute_proxy_alpha(self.gate_proj.layer_idx, x)
            predicted = mgr.predictor.predict_experts(
                self.gate_proj.layer_idx, alpha, top_k=getattr(mgr, "prefetch_top_k", 2)
            )
            mgr.prefetch(self.gate_proj.layer_idx, predicted)
            self._last_alpha = alpha

        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        gate_tensors, up_tensors, down_tensors, remapped = (
            mgr.prepare_gather_triple_quantized(
                self.gate_proj.layer_idx, idx, mode="gather"
            )
        )

        gate_w, gate_s, gate_b = gate_tensors
        up_w, up_s, up_b = up_tensors
        down_w, down_s, down_b = down_tensors

        x_up = mx.gather_qmm(
            x,
            up_w,
            up_s,
            up_b,
            rhs_indices=remapped,
            transpose=True,
            group_size=self.up_proj.group_size,
            bits=self.up_proj.bits,
            mode=self.up_proj.mode,
            sorted_indices=do_sort,
        )
        if "bias" in self.up_proj:
            x_up = x_up + mx.expand_dims(self.up_proj["bias"][idx], -2)

        x_gate = mx.gather_qmm(
            x,
            gate_w,
            gate_s,
            gate_b,
            rhs_indices=remapped,
            transpose=True,
            group_size=self.gate_proj.group_size,
            bits=self.gate_proj.bits,
            mode=self.gate_proj.mode,
            sorted_indices=do_sort,
        )
        if "bias" in self.gate_proj:
            x_gate = x_gate + mx.expand_dims(self.gate_proj["bias"][idx], -2)

        x = mx.gather_qmm(
            self.activation(x_up, x_gate),
            down_w,
            down_s,
            down_b,
            rhs_indices=remapped,
            transpose=True,
            group_size=self.down_proj.group_size,
            bits=self.down_proj.bits,
            mode=self.down_proj.mode,
            sorted_indices=do_sort,
        )
        if "bias" in self.down_proj:
            x = x + mx.expand_dims(self.down_proj["bias"][idx], -2)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        out = x.squeeze(-2)
        mx.eval(out)

        if mgr.predictor is not None and hasattr(self, "_last_alpha"):
            mgr.predictor.record_activation(
                self.gate_proj.layer_idx, self._last_alpha, idx
            )

        return out
