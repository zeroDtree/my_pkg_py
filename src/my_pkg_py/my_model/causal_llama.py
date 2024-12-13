from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from my_utils import HF_MIRROR, add_maybe_special_tokens
from transformers import LlamaConfig
from typing import Tuple


def get_causal_llama_model(
    pretrained=False,
    model_name="meta-llama/Llama-2-7b-hf",
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=1,
    num_key_value_heads=None,
    hidden_act="silu",
    max_position_embeddings=2048,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=True,
    pad_token_id=None,
    bos_token_id=1,
    eos_token_id=2,
    pretraining_tp=1,
    tie_word_embeddings=False,
    rope_theta=10000.0,
    rope_scaling=None,
    attention_bias=False,
    attention_dropout=0.0,
    mlp_bias=False,
    tokenizer=None,
) -> Tuple:
    HF_MIRROR.set_hf_mirror()

    if not pretrained:
        model_config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pretraining_tp=pretraining_tp,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            mlp_bias=mlp_bias,
        )
        model = AutoModelForCausalLM.from_config(model_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model, tokenizer = add_maybe_special_tokens(model, tokenizer)
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = get_causal_llama_model(num_hidden_layers=1, pretrained=True)
    print(model)
    print(tokenizer)
