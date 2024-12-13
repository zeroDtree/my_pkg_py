import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from my_utils import add_maybe_special_tokens
from .causal_llama import get_causal_llama_model
from .decoder_tf import get_zls_causal_model


def get_text_to_text_model(
    model_name: str,
    model_type: str = "CausalLM",
    dtype: str = "bf16",
    tokenizer: str = None,
    flash_attention: bool = False,
):
    assert model_type in ["CausalLM", "ConditionalGeneration"]
    match model_type:
        case "CausalLM":
            auto_model_class = AutoModelForCausalLM
        case "ConditionalGeneration":
            auto_model_class = AutoModelForSeq2SeqLM
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")
    model_config = dict(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
    )
    if flash_attention:
        model_config["attn_implementation"] = "flash_attention_2"
    match dtype:
        case "fp32":
            model_config["torch_dtype"] = torch.float32
        case "bf16":
            model_config["torch_dtype"] = torch.bfloat16
        case "int8":
            quant_8bit_config = dict(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            model_config["quantization_config"] = quant_8bit_config
        case "nf4":
            quant_4bit_config = dict(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_config["quantization_config"] = quant_4bit_config
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")
    model = auto_model_class.from_pretrained(**model_config)
    if tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model, tokenizer = add_maybe_special_tokens(model, tokenizer)
    return model, tokenizer


def get_model_and_tokenizer(model_name, pretrained=False):
    if pretrained:
        model_name = model_name
        model, tokenizer = get_text_to_text_model(model_name)
    else:
        match model_name:
            case "causal_llama":
                model, tokenizer = get_causal_llama_model(
                    pretrained=False, num_hidden_layers=1
                )
            case "zls_causal_model":
                tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
                model = get_zls_causal_model(
                    vocab_size=tokenizer.vocab_size,
                    embed_dim=512,
                    num_head=8,
                    dropout=0,
                    num_block=16,
                    max_pos_len=5000,
                    batch_first=True,
                )
                model, tokenizer = add_maybe_special_tokens(model, tokenizer)
            case _:
                raise ValueError(f"Unsupported model type: {model_name}")
    return model, tokenizer
