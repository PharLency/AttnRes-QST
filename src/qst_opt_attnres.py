"""AttnRes-QST: drop-in extensions of the QST classes with Attention Residuals.

This module **does not modify** `third_party/qst/modeling_opt_qst.py`. It only
subclasses the original QST classes and injects AttnRes aggregation at the
inter-layer z-gate (see `src/qst_arch_notes.md` §4.1 for the insertion point).

When `AttnResConfig.use_attn_residual=False`, behaviour is identical to the
original QST — `forward` delegates to `super().forward`, no AttnRes params are
allocated, and RNG is not consumed, so baseline runs are reproducible.

Class hierarchy
---------------
    QSTOPTDecoderAttnRes(QSTOPTDecoder)
        └── overrides __init__ to (optionally) add attn_residual_modules
        └── overrides forward: when enabled, maintains a running qst_history
            and replaces the `z * qst_hidden_states` carry term with
            `z * AttnRes(qst_history)`.

    QSTOPTModelAttnRes(QSTOPTModel)
        └── swaps in QSTOPTDecoderAttnRes as `self.decoder`.

    QSTOPTForSequenceClassificationAttnRes(QSTOPTForSequenceClassification)
        └── swaps in QSTOPTModelAttnRes as `self.model`.

Gradient checkpointing is not implemented on the AttnRes path (4-bit + bf16
training on OPT-1.3B with our side-network size does not strictly need it;
flip it off or disable AttnRes to train with gc).
"""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Make the original QST files importable (they live under third_party/).
_QST_DIR = Path(__file__).resolve().parent.parent / "third_party" / "qst"
if str(_QST_DIR) not in sys.path:
    sys.path.insert(0, str(_QST_DIR))

from modeling_opt_qst import (  # noqa: E402
    QSTOPTDecoder,
    QSTOPTModel,
    QSTOPTForSequenceClassification,
)
from modeling_qst_output import QSTBaseModelOutputWithPast  # noqa: E402

from attn_residual import AttentionResidual, BlockAttentionResidual  # noqa: E402


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class AttnResConfig:
    """Controls the AttnRes injection into QST's side network.

    Attributes
    ----------
    use_attn_residual : bool
        If False, every class in this module behaves identically to the QST
        baseline (and allocates no AttnRes params).
    attn_residual_type : {"full", "block"}
        "full"  — aggregate the entire side-state history
        "block" — aggregate only the last `block_size` states (sliding window)
    block_size : int
        Used only when attn_residual_type == "block".
    attn_residual_norm : {"rmsnorm", "layernorm", "none"}
        Normalisation applied inside the AttnRes pseudo-query dot product.
    freeze_query : bool
        If True, the pseudo-query is frozen at zero. Softmax(0,...,0) = uniform,
        so AttnRes degenerates into a plain unweighted mean of side-layer
        hidden states. Used as an ablation to verify gains come from learnable
        attention rather than simple averaging.
    """
    use_attn_residual: bool = False
    attn_residual_type: str = "block"
    block_size: int = 2
    attn_residual_norm: str = "rmsnorm"
    freeze_query: bool = False


# --------------------------------------------------------------------------- #
#  Decoder
# --------------------------------------------------------------------------- #
class QSTOPTDecoderAttnRes(QSTOPTDecoder):
    """QSTOPTDecoder with Attention Residual aggregation between side layers."""

    def __init__(self, llm, config, qst_config, hf_device_map, attn_res_config: AttnResConfig):
        super().__init__(llm, config, qst_config, hf_device_map)
        self.attn_res_config = attn_res_config

        if not attn_res_config.use_attn_residual:
            self.attn_residual_modules = None
            return

        side_dim = int(config.hidden_size / qst_config.r)
        mods = []
        kind = attn_res_config.attn_residual_type.lower()
        for _ in range(config.num_hidden_layers):
            if kind == "full":
                mods.append(AttentionResidual(
                    hidden_dim=side_dim,
                    norm_type=attn_res_config.attn_residual_norm,
                    freeze_query=attn_res_config.freeze_query,
                ))
            elif kind == "block":
                mods.append(BlockAttentionResidual(
                    hidden_dim=side_dim,
                    block_size=attn_res_config.block_size,
                    norm_type=attn_res_config.attn_residual_norm,
                    freeze_query=attn_res_config.freeze_query,
                ))
            else:
                raise ValueError(f"Unknown attn_residual_type: {kind!r}")
        self.attn_residual_modules = nn.ModuleList(mods)

        # Place each AttnRes module on the same device as its corresponding
        # qst_layer (this matters under multi-GPU device_map="auto").
        for i in range(config.num_hidden_layers):
            try:
                dev = next(self.qst_layers[i].parameters()).device
            except StopIteration:
                dev = torch.device("cpu")
            self.attn_residual_modules[i] = self.attn_residual_modules[i].to(dev)

    # -------- forward ------------------------------------------------------- #
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            qst_past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_qst_attentions=False,
            output_qst_hidden_states=False,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QSTBaseModelOutputWithPast]:
        # Baseline short-circuit: no AttnRes → identical to original QST.
        if not self.attn_res_config.use_attn_residual:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_values=past_key_values,
                qst_past_key_values=qst_past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_qst_attentions=output_qst_attentions,
                output_qst_hidden_states=output_qst_hidden_states,
                return_dict=return_dict,
            )

        # --- AttnRes path. Faithfully duplicates QSTOPTDecoder.forward
        # (third_party/qst/modeling_opt_qst.py L991–L1259), except for the
        # two AttnRes-flagged blocks marked below. ---
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        with torch.no_grad():
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        mask_seq_length = past_key_values_length + seq_length

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        elif attention_mask.shape[1] != mask_seq_length:
            raise ValueError(
                f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                f"{mask_seq_length} (sum of the lengths of current and past inputs)"
            )
        causal_attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        with torch.no_grad():
            pos_embeds = self.embed_positions(attention_mask, past_key_values_length)
            if self.project_in is not None:
                inputs_embeds = self.project_in(inputs_embeds)
            hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            raise NotImplementedError(
                "gradient_checkpointing is not supported with use_attn_residual=True. "
                "Either disable gc or set use_attn_residual=False."
            )

        all_hidden_states = () if output_hidden_states else None
        qst_all_hidden_states = () if output_qst_hidden_states else None
        all_self_attns = () if output_attentions else None
        qst_self_attns = () if output_qst_attentions else None

        if use_cache:
            next_decoder_cache = ()
            qst_next_decoder_cache = ()
        else:
            next_decoder_cache = None
            qst_next_decoder_cache = None

        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None and attn_mask.size()[0] != len(self.backbone):
                raise ValueError(
                    f"The `{mask_name}` should be specified for {len(self.backbone)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        qst_hidden_states = self.downsample[0](hidden_states)

        # ==================== [AttnRes] history bookkeeping ==================== #
        # Running list of side-layer outputs h^side_0, h^side_1, ... that
        # AttnRes aggregates to replace the `z * qst_hidden_states` carry term.
        qst_history: List[torch.Tensor] = []
        # ======================================================================= #

        for idx, decoder_layer in enumerate(self.backbone):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                qst_all_hidden_states += (qst_hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            qst_past_key_value = qst_past_key_values[idx] if qst_past_key_values is not None else None

            with torch.no_grad():
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            z = torch.sigmoid(self.z[idx])
            qst_hidden_states = qst_hidden_states.to(hidden_states.device)

            # ================ [AttnRes] replace z-gated carry term ================ #
            # Original QST uses `z * qst_hidden_states` (i.e. only h^side_{l-1}).
            # AttnRes aggregates the full history (Full) or a sliding window (Block).
            # At idx=0 there is no history; fall back to the init projection.
            if len(qst_history) >= 1:
                # AttnRes output sits on the same device as its params; ensure
                # it matches qst_hidden_states' device before the combination.
                h_carry = self.attn_residual_modules[idx](qst_history)
                h_carry = h_carry.to(qst_hidden_states.device)
            else:
                h_carry = qst_hidden_states
            # ====================================================================== #

            qst_hidden_states = (1 - z) * self.downsample[idx](hidden_states) + z * h_carry

            qst_layer_outputs = self.qst_layers[idx](
                qst_hidden_states,
                attention_mask=causal_attention_mask.to(qst_hidden_states.device),
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                past_key_value=qst_past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            qst_hidden_states = qst_layer_outputs[0]

            # [AttnRes] append this layer's output to history for future layers.
            qst_history.append(qst_hidden_states)

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
                qst_next_decoder_cache += (qst_layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if output_qst_attentions:
                qst_self_attns += (qst_layer_outputs[1],)

        with torch.no_grad():
            if self.final_layer_norm is not None:
                hidden_states = self.final_layer_norm(hidden_states.to(self.final_layer_norm.weight.device))
            if self.project_out is not None:
                hidden_states = self.project_out(hidden_states)

        if self.final_layer_norm_qst is not None:
            qst_hidden_states = self.final_layer_norm_qst(
                qst_hidden_states.to(self.final_layer_norm.weight.device)
            )

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if output_qst_attentions:
            qst_all_hidden_states += (qst_hidden_states,)

        if use_cache:
            next_cache = next_decoder_cache
            qst_next_cache = qst_next_decoder_cache
        else:
            next_cache = None
            qst_next_cache = None

        if not return_dict:
            return tuple(v for v in [
                qst_hidden_states, hidden_states, qst_next_cache, next_cache,
                qst_all_hidden_states, all_hidden_states, qst_self_attns, all_self_attns,
            ] if v is not None)

        return QSTBaseModelOutputWithPast(
            last_qst_hidden_states=qst_hidden_states,
            last_hidden_states=hidden_states,
            qst_past_key_values=qst_next_cache,
            past_key_values=next_cache,
            qst_hidden_states=qst_all_hidden_states,
            hidden_states=all_hidden_states,
            qst_attentions=qst_self_attns,
            attentions=all_self_attns,
        )


# --------------------------------------------------------------------------- #
#  Model + SeqCls wrappers (just swap in our decoder / model)
# --------------------------------------------------------------------------- #
class QSTOPTModelAttnRes(QSTOPTModel):
    def __init__(self, llm, config, qst_config, hf_device_map, attn_res_config: AttnResConfig):
        # super's __init__ builds a baseline QSTOPTDecoder; we overwrite it.
        # The baseline instance is transient and GC'd once replaced.
        super().__init__(llm, config, qst_config, hf_device_map)
        self.decoder = QSTOPTDecoderAttnRes(
            llm.decoder, config, qst_config, hf_device_map, attn_res_config,
        )


class QSTOPTForSequenceClassificationAttnRes(QSTOPTForSequenceClassification):
    def __init__(self, llm, config, qst_config, attn_res_config: AttnResConfig):
        # Fast path: when AttnRes is disabled we do **no** extra construction
        # after super(). This keeps RNG consumption identical to the plain
        # baseline class so sanity checks (same-seed → same weights → same
        # outputs) hold bit-for-bit.
        if not attn_res_config.use_attn_residual:
            super().__init__(llm, config, qst_config)
            self.attn_res_config = attn_res_config
            return

        # Hold refs we still need after super's `del llm`.
        llm_model = llm.model
        llm_hf_map = getattr(llm, "hf_device_map", {}) or {}

        # super builds baseline QSTOPTModel + score_z + upsample + score + device moves.
        super().__init__(llm, config, qst_config)

        # Replace the baseline sub-model with our AttnRes-aware variant.
        self.model = QSTOPTModelAttnRes(
            llm_model, config, qst_config, llm_hf_map, attn_res_config,
        )
        self.hf_device_map = self.model.decoder.hf_device_map
        self.attn_res_config = attn_res_config
