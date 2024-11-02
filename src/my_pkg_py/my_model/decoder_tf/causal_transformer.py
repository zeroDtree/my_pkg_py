import torch
from typing import Optional, Union, List
import math
from torch import Tensor, Tuple

ACTIVATION_MAP = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
}


class FeedForwardBlock(torch.nn.Module):
    def __init__(self, embed_dim, k=4, dropout=0.0, bias=False, act="relu"):
        super().__init__()
        self.linear_1 = torch.nn.Linear(embed_dim, k * embed_dim, bias=bias)
        self.act = ACTIVATION_MAP[act]()
        self.linear_2 = torch.nn.Linear(k * embed_dim, embed_dim, bias=bias)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(torch.nn.Module):
    
    def __init__(self,
            embed_dim,
            num_heads,
            dropout=0,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=True,
            device=None,
            dtype=None,
        ):

        super().__init__()
        self.d_model = embed_dim
        self.d_head = embed_dim // num_heads
        self.num_heads = num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.q_linear = torch.nn.Linear(embed_dim, self.kdim, bias=bias)
        self.k_linear = torch.nn.Linear(embed_dim, self.kdim, bias=bias)
        self.v_linear = torch.nn.Linear(embed_dim, self.vdim, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.out_linear = torch.nn.Linear(self.vdim, embed_dim, bias=bias)

    def merge_masks(
        self,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        batch_size: int,
        seq_len: int,
    ) -> Tuple[Optional[Tensor], Optional[int]]:
        r"""Determine mask type and combine masks if necessary.

        If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        if key_padding_mask is not None:
            mask_type = 1
            # (batch_size, seq_L)
            merged_mask = key_padding_mask

        if attn_mask is not None:
            mask_type = 2

            # Always expands attn_mask to 4D
            if attn_mask.dim() == 3:  # (batch_size, seq_L, seq_L)
                # (batch_size, seq_L, seq_L) -> (batch_size, 1, seq_L, seq_L)
                attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2: #(seq_L, seq_L)
                # (seq_L, seq_L) -> (1, 1, seq_L, seq_L) -> (batch_size, head_num, seq_L, seq_L)
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(
                    batch_size, self.num_heads, -1, -1
                )
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                # (bs, seq_L) -> (bs, 1, 1, seq_L) -> （batch_size, head_num, 1, seq_L）
                key_padding_mask_expanded = key_padding_mask.view(
                    batch_size, 1, 1, seq_len
                ).expand(-1, self.num_heads, -1, -1)
                # (bs, 1 or head_num, seq_L, seq_L) + (bs, head_num, 1, seq_L) -> (bs, head_num, seq_L, seq_L)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded
        # no attn_mask and no key_padding_mask, returns None, None
        return merged_mask, mask_type

    def attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        mask, mask_type = self.merge_masks(
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            batch_size=query.shape[0],
            seq_len=query.shape[-2],
        )
        mask = mask.to(device=query.device, dtype=query.dtype)
        # (bs, head_num, seq_L, kdim) @ (bs, head_num, kdim, seq_L) -> (bs, head_num, seq_L, seq_L)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(attn_mask != 0.0, float("-inf"))
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        # (bs, head_num, seq_L, seq_L) @ (bs, head_num, seq_L, vdim) -> (bs, head_num, seq_L, vdim)
        output = torch.matmul(scores, value)
        if need_weights:
            if average_attn_weights:
                scores = torch.mean(scores, dim=1)
            return output, scores
        else:
            return output

    def forward(
        self,
        q,
        k,
        v,
        key_padding_mask=None,
        attn_mask=None,
        average_attn_weights=True,
        need_weights=True,
        is_causal=False,
    ):
        # q,k,v size(bs, seq_L, d_model)
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_head)
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_head)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_head)
        # bs, seq_L, head_num, head_dim）-> (bs, head_num, seq_L, head_dim)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        att = self.attention(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
        if need_weights:
            x, att_weight = att
        x = x.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        x = self.out_linear(x)
        if need_weights:
            return x, att_weight
        else:
            return x


class AttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0, batch_first=True):
        super().__init__()
        self.att = MultiHeadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=embed_dim,
            vdim=embed_dim,
            batch_first=batch_first,
            device=None,
            dtype=None,
        )
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(
        self,
        x,
        att_mask=None,
        key_padding_mask=None,
        average_attn_weights=True,
        is_causal=True,
        past_key_values=None,
        use_cache=False,
    ):
        # x.shape = (batch_size, seq_len, embed_dim)
        if past_key_values is not None:
            if use_cache:
                pass
        x, att_weight = self.att(
            q=x,
            k=x,
            v=x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=att_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        x = self.dropout(x)
        return {"x": x, "att_weight": att_weight}


class TransformerBlock(torch.nn.Module):
    def __init__(self, embed_dim, num_head, dropout, batch_first=True):
        super().__init__()
        self.att = AttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_head,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.ff = FeedForwardBlock(embed_dim=embed_dim, k=4, dropout=dropout)

    def forward(
        self,
        x,
        att_mask=None,
        key_padding_mask=None,
        average_attn_weights=True,
        is_causal=True,
    ):
        x_residual = x
        x = self.att(
            x,
            att_mask=att_mask,
            key_padding_mask=key_padding_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        x, att_weight = x["x"], x["att_weight"]
        x = x_residual + x
        x_residual = x
        x = self.ff(x)
        x = x_residual + x
        return {"x": x, "att_weight": att_weight}


class CausalLanguageModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_head,
        dropout=0,
        num_block=3,
        max_pos_len=5000,
        batch_first=True,
    ):
        super().__init__()
        self.wte = torch.nn.Embedding(vocab_size, embed_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_head=num_head,
                    dropout=dropout,
                    batch_first=batch_first,
                )
                for i in range(num_block)
            ]
        )

    def generate_square_subsequent_mask(self, sz: int, device=None, dtype=None):
        r"""Generate a square causal mask for the sequence.

        The masked positions are filled with 'True'. Unmasked positions are filled with False
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.bool
        return torch.triu(torch.ones(sz, sz, device=device, dtype=dtype), diagonal=1)

    def forward(
        self,
        x: torch.Tensor,
        att_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        is_causal: bool = True,
    ):
        if is_causal and att_mask is None:
            att_mask = self.generate_square_subsequent_mask(x.size(1), device=x.device)
        att_weight_list = []
        x = self.wte(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(
                x,
                att_mask=att_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
            )
            att_weight_list.append(x["att_weight"])
            x = x["x"]
        return {"x": x, "att_weight": att_weight_list}


class CausalLanguageModelConfig:
    def __init__(
        self,
        vocab_size=32000,
        embed_dim=1024,
        num_head=2,
        dropout=0,
        num_block=3,
        max_pos_len=5000,
        batch_first=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.dropout = dropout
        self.num_block = num_block
        self.max_pos_len = max_pos_len
        self.batch_first = batch_first
        self.kwargs = kwargs


from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class CausalLanguageModelConfigForAuto(PretrainedConfig):
    model_type = "decoder-only-transformer"

    def __init__(
        self,
        vocab_size=30000,
        embed_dim=1024,
        num_head=2,
        dropout=0,
        num_block=3,
        max_pos_len=5000,
        batch_first=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.dropout = dropout
        self.num_block = num_block
        self.max_pos_len = max_pos_len
        self.batch_first = batch_first


class CausalLanguageModelForAuto(PreTrainedModel):
    config_class = CausalLanguageModelConfigForAuto
    base_model_prefix = "zls_causal_tf"

    def __init__(self, config: CausalLanguageModelConfigForAuto):
        super().__init__(config)
        self.model = CausalLanguageModel(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_head=config.num_head,
            dropout=config.dropout,
            num_block=config.num_block,
            max_pos_len=config.max_pos_len,
            batch_first=config.batch_first,
        )
        self.lm_head = torch.nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        # Adjust the forward method to match the expected input/output format
        outputs = self.model(
            input_ids,
            att_mask=None,
            key_padding_mask=(
                ~attention_mask.bool() if attention_mask is not None else None
            ),
        )
        logits = self.lm_head(outputs["x"])
        loss = None
        if labels is not None:
            # shift logits and labels for computing the loss
            # shape = (batch_size, seq_length, vocab_size)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            # attentions=outputs["att_weight"],
            cross_attentions=None,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, **kwargs
    ):
        # This method prepares inputs for the generate method
        input_shape = input_ids.shape
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
        }

    def get_input_embeddings(self):
        return self.model.wte

    def get_output_embeddings(self):
        return self.lm_head


def register_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    model_name = CausalLanguageModelConfigForAuto.model_type
    AutoConfig.register(model_name, CausalLanguageModelConfigForAuto)
    AutoModelForCausalLM.register(
        CausalLanguageModelConfigForAuto, CausalLanguageModelForAuto
    )


if __name__ == "__main__":

    # test_config = CausalLanguageModelConfigForAuto(
    #     vocab_size=1000,
    #     embed_dim=256,
    #     num_head=4,
    #     dropout=0.1,
    #     num_block=2,
    #     max_pos_len=1000,
    #     batch_first=True,
    # )
    register_model()
    from transformers import AutoConfig, AutoModelForCausalLM

    test_config = AutoConfig.from_pretrained("zengls/decoder-tf")

    # print("Test Configuration:")
    # print(test_config)
    # test_model = AutoModelForCausalLM.from_config(test_config)
    test_model = AutoModelForCausalLM.from_pretrained("zengls/decoder-tf")

    print("\nModel structure:")
    print(test_model)

    # Test the forward pass
    import torch

    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, test_config.vocab_size, size=(batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids).bool()

    outputs = test_model(input_ids, attention_mask=attention_mask, labels=input_ids)

    print(f"outputs: {outputs}")
    print("\nOutput shape:")
    print(outputs["logits"].shape)

    # Expected shape: (batch_size, seq_length, vocab_size)
    assert outputs["logits"].shape == (
        batch_size,
        seq_length,
        test_config.vocab_size,
    ), "Output shape mismatch"

    print("\nTest passed successfully!")

    for name, param in test_model.named_parameters():
        print(name)

    # test_model.save_pretrained(
    #     "test_model", repo_id="zengls/decoder-tf", push_to_hub=True
    # )
    # test_config.save_pretrained(
    #     "test_config", repo_id="zengls/decoder-tf", push_to_hub=True
    # )

    # model = AutoModelForCausalLM.from_pretrained("zengls/decoder-tf")
    # print(model)
    input_ids = torch.randint(0, test_config.vocab_size, size=(1, 10))
    generated_ids = test_model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    print(generated_ids)
    print("Generated sequence shape:", generated_ids.shape)
