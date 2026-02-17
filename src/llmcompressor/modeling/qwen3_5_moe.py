"""
Qwen3.5-MoE calibration module.

Qwen3.5-MoE structure is nearly identical to Qwen3Next:
- Same forward logic with shared_expert and shared_expert_gate
- Same gate returning (router_logits, router_scores, router_indices)
- Same experts structure (3D packed weights with optimized routing)

This creates a separate experts wrapper using the same pattern as Qwen3VLMoe.
"""
from typing import TYPE_CHECKING

import torch

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize

if TYPE_CHECKING:
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeSparseMoeBlock,
    )


@MoECalibrationModule.register("Qwen3_5MoeSparseMoeBlock")
class CalibrationQwen3_5MoeSparseMoeBlock(MoECalibrationModule):
    """
    Calibration version of Qwen3_5MoeSparseMoeBlock that sends all tokens to all experts.

    During calibration, when calibrate_all_experts=True, all tokens are sent to all
    experts to ensure proper quantization statistics are collected for every expert,
    not just those activated by the calibration data routing.

    Structure:
    - Uses shared_expert (always activated) for mixing
    - Top-k router for sparse expert selection
    - Experts use 3D packed weights (num_experts, in_dim, out_dim/2)
    """

    is_permanent = False

    def __init__(
        self,
        original: "Qwen3_5MoeSparseMoeBlock",
        config: "Qwen3_5MoeConfig",
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.hidden_size = original.gate.weight.shape[1]
        self.num_experts = original.gate.weight.shape[0]
        self.top_k = getattr(original.gate, "top_k", 2)
        self.norm_topk_prob = True

        self.calibrate_all_experts = calibrate_all_experts
        self.gate = original.gate
        self.experts = SequentialQwen3_5MoeTextExperts(config, original.experts)
        self.shared_expert = original.shared_expert
        self.shared_expert_gate = original.shared_expert_gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional calibration mode.

        When calibrate_all_experts=True:
            - All tokens are sent to all experts for calibration
            - Routing weights are still used for final output combination
            - This ensures all experts see calibration data
        When calibrate_all_experts=False:
            - Normal MoE routing behavior (only routed tokens go to each expert)
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # gate returns: (router_logits, router_scores, router_indices)
        _, routing_weights, selected_experts = self.gate(hidden_states)

        # normalized routing weights (already normalized in gate)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Process shared expert (always active)
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )

        # One hot encode the selected experts to create an expert mask
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx, expert_layer in enumerate(self.experts):
            idx, token_idx = torch.where(expert_mask[expert_idx].squeeze(0))

            if self.calibrate_all_experts:
                # Send all tokens to the expert for calibration, use only routed tokens for output
                expert_out = expert_layer(hidden_states)[token_idx]
            else:
                if len(token_idx) > 0:
                    expert_out = expert_layer(hidden_states[token_idx])
                else:
                    continue

            if len(token_idx) > 0:
                weighted_output = expert_out * routing_weights[token_idx, idx, None]
                final_hidden_states.index_add_(
                    0, token_idx, weighted_output.to(hidden_states.dtype)
                )

        # Combine sparse expert output with shared expert output
        expert_output = final_hidden_states + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)

        return expert_output

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        """
        Restore the original module structure.

        Since is_permanent=False, this method is called when exiting
        the calibration context to restore the original MoE module.
        """
        return original


class SequentialQwen3_5MoeTextExperts(torch.nn.ModuleList):
    """
    Wrapper class for Qwen3.5-MoE experts that makes them iterable.

    This extracts the 3D packed weights from Qwen3_5MoeExperts and
    creates individual MLP modules that can be iterated during calibration.
    Follows the same pattern as SequentialQwen3VLMoeTextExperts.
    """

    def __init__(self, config: "Qwen3_5MoeConfig", original):
        # Create simple MLP wrappers without depending on transformers MLP class
        self.num_experts = original.gate_up_proj.shape[0]
        with skip_weights_initialize():
            super().__init__()
            for i in range(self.num_experts):
                mlp = _create_qwen3_5_moe_mlp_wrapper(original, i, config)
                self.append(mlp)


def _create_qwen3_5_moe_mlp_wrapper(original, expert_idx, config):
    """
    Create a simple MLP wrapper for a single expert without modifying transformers code.
    """
    import torch.nn as nn

    # Extract weights for this expert
    # original.gate_up_proj shape: (num_experts, in_dim, intermediate_size * 2)
    # original.down_proj shape: (num_experts, intermediate_size, out_dim)
    gate_up = original.gate_up_proj[expert_idx]
    down = original.down_proj[expert_idx]

    intermediate_size = gate_up.shape[1] // 2
    hidden_dim = gate_up.shape[0]
    out_dim = down.shape[1]

    # Create Linear layers with dummy weights (will be replaced)
    gate_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
    up_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
    down_proj = nn.Linear(intermediate_size, out_dim, bias=False)

    # Copy the weights from the packed representation
    gate_proj.weight.data = gate_up[:, :intermediate_size].t().clone().contiguous()
    up_proj.weight.data = gate_up[:, intermediate_size:].t().clone().contiguous()
    down_proj.weight.data = down.t().clone().contiguous()

    # Wrap in a module that matches MLP forward signature
    from transformers.activations import SiLUActivation

    class Qwen3_5MoeMLPWrapper(nn.Module):
        def __init__(self, gate_proj, up_proj, down_proj):
            super().__init__()
            self.gate_proj = gate_proj
            self.up_proj = up_proj
            self.down_proj = down_proj
            self.act_fn = SiLUActivation()

        def forward(self, x):
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            return down_proj

    return Qwen3_5MoeMLPWrapper(gate_proj, up_proj, down_proj)