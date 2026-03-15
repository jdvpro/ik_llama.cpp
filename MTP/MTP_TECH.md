
# Техническая информация

## Model Overview

Qwen3-Next-80B-A3B-Instruct has the following features:

### Основные характеристики

- **Type:** Causal Language Models
- **Training Stage:** Pretraining (15T tokens) & Post-training
- **Number of Parameters:** 80B in total and 3B activated
- **Number of Parameters (Non-Embedding):** 79B
- **Hidden Dimension:** 2048
- **Number of Layers:** 48
    - Hybrid Layout: 12 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))

### Gated Attention

- **Number of Attention Heads:** 16 for Q and 2 for KV
- **Head Dimension:** 256
- **Rotary Position Embedding Dimension:** 64

### Gated DeltaNet

- **Number of Linear Attention Heads:** 32 for V and 16 for QK
- **Head Dimension:** 128

### Mixture of Experts

- **Number of Experts:** 512
- **Number of Activated Experts:** 10
- **Number of Shared Experts:** 1
- **Expert Intermediate Dimension:** 512

### Context Length

- 262,144 natively and extensible up to 1,010,000 tokens



## Dump MTP tensors in gguf format
```
844:    8388608 |  4096,  2048,     1,     1 | F16     | blk.48.nextn.eh_proj.weight
845:       2048 |  2048,     1,     1,     1 | F32     | blk.48.attn_norm.weight
846:  536870912 |   512,  2048,   512,     1 | F16     | blk.48.ffn_down_exps.weight
847:  536870912 |  2048,   512,   512,     1 | F16     | blk.48.ffn_gate_exps.weight
848:  536870912 |  2048,   512,   512,     1 | F16     | blk.48.ffn_up_exps.weight
849:    1048576 |  2048,   512,     1,     1 | F16     | blk.48.ffn_gate_inp.weight
850:    1048576 |   512,  2048,     1,     1 | F16     | blk.48.ffn_down_shexp.weight
851:    1048576 |  2048,   512,     1,     1 | F16     | blk.48.ffn_gate_shexp.weight
852:    1048576 |  2048,   512,     1,     1 | F16     | blk.48.ffn_up_shexp.weight
853:       2048 |  2048,     1,     1,     1 | F16     | blk.48.ffn_gate_inp_shexp.weight
854:       2048 |  2048,     1,     1,     1 | F32     | blk.48.post_attention_norm.weight
855:        256 |   256,     1,     1,     1 | F32     | blk.48.attn_k_norm.weight
856:    1048576 |  2048,   512,     1,     1 | F16     | blk.48.attn_k.weight
857:    8388608 |  4096,  2048,     1,     1 | F16     | blk.48.attn_output.weight
858:        256 |   256,     1,     1,     1 | F32     | blk.48.attn_q_norm.weight
859:   16777216 |  2048,  8192,     1,     1 | F16     | blk.48.attn_q.weight
860:    1048576 |  2048,   512,     1,     1 | F16     | blk.48.attn_v.weight
861:       2048 |  2048,     1,     1,     1 | F32     | blk.48.nextn.shared_head_norm.weight
862:       2048 |  2048,     1,     1,     1 | F32     | blk.48.nextn.enorm.weight
863:       2048 |  2048,     1,     1,     1 | F32     | blk.48.nextn.hnorm.weight
```


## Пример реализации из vLLM

```python

@support_torch_compile
class Qwen3NextMultiTokenPredictor(nn.Module):
def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
super().__init__()

        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config

        config: Qwen3NextConfig = model_config.hf_config

        self.config = config

        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        self.fc = ColumnParallelLinear(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            gather_output=True,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc",
        )

        self.layers = torch.nn.ModuleList(
            Qwen3NextDecoderLayer(
                vllm_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_embedding = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
            hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
            hidden_states = self.fc(hidden_states)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        current_step_idx = spec_step_idx % self.num_mtp_layers
        hidden_states, residual = self.layers[current_step_idx](
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

```

**Важно:** `shared` — `tok_embd` и `output` (lm_head) это те же тензоры, что у основной модели, не копировать, не создавать новые.
