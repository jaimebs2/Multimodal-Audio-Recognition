from CustomQWen import audio_encoder
from CustomQWen.qwenblock import QWenBlock, RotaryEmbedding, RMSNorm
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
import torch.nn as nn
import math
from torch.nn import CrossEntropyLoss

class Model():

    def __init__(self,   
                 model,
                 device,
                 config,
                 qwen_config,
                 vocab_size=155947,
                 embed_dim=4096,
                 seq_length=2048,
                 audio_start_id=155163,
                 ):

        self.device = device
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.audio_start_id = audio_start_id

        self.wte = nn.Embedding(vocab_size, embed_dim).bfloat16().to(device)
        self.wte.load_state_dict(model.transformer.wte.state_dict())

        self.encoder = audio_encoder.AudioEncoder(**config.audio)
        self.encoder.load_state_dict(model.transformer.audio.state_dict())
        self.encoder.bfloat16().to(self.device)
        #self.encoder.eval()

        self.rotary_emb = RotaryEmbedding(dim=128, base=10000).to(self.device)
        self.rotary_emb.load_state_dict(model.transformer.rotary_emb.state_dict())

        self.drop = nn.Dropout(0.0).bfloat16().to(self.device)
        self.h = nn.ModuleList(
                    [
                        QWenBlock(
                            qwen_config
                        )
                        for i in range(32)
                    ]
                ).bfloat16().to(self.device)
        self.h.load_state_dict(model.transformer.h.state_dict())

        self.ln_f = RMSNorm(
                self.embed_dim,
                eps=1e-06,
            ).bfloat16().to(self.device)

        self.ln_f.bfloat16().load_state_dict(model.transformer.ln_f.state_dict())

        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size).bfloat16().to(self.device)
        self.lm_head.weight = model.lm_head.weight
        self.lm_head.bias = None
        self.lm_head.to(self.device)

    def get_ntk_alpha(
            self, 
            true_seq_len
            ):
        context_value = math.log(true_seq_len / self.seq_length, 2) + 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)
        return ntk_alpha

    def forward(
            self,
            input_ids,
            position_ids,
            attention_mask,
            token_type_ids,
            audio_info,
            past_key_values=None,
            use_cache=True,
            output_attentions=False,
            labels = None,
            ):

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
        past_length = 0
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=self.device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Audio Encoder
        audios = audio_info["input_audios"]
        audio_span_tokens = audio_info["audio_span_tokens"]
        input_audio_lengths = audio_info["input_audio_lengths"]

        bos_pos = torch.where(input_ids == self.audio_start_id)
        eos_pos = torch.where(input_ids == self.audio_start_id + 1)
        assert (bos_pos[0] == eos_pos[0]).all()
        audio_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)

        with torch.no_grad():
            real_input_audio_lens = input_audio_lengths[:, 0].tolist()
            max_len_in_batch = max(real_input_audio_lens)
            padding_mask = torch.ones([audios.size(0), max_len_in_batch]).to(dtype=torch.bfloat16,
                                                                                    device=self.device)
            for index in range(len(audios)):
                padding_mask[index, :input_audio_lengths[index][0].item()] = 0
            x, bos, eos = self.encoder(audios, padding_mask,input_audio_lengths)
            output_audios = []
            for i in range(len(audio_span_tokens)):
                audio_span = audio_span_tokens[i]
                audio = x[i][:audio_span-2]
                if bos is not None:
                    audio = torch.concat([bos, audio, eos])
                assert len(audio) == audio_span
                output_audios.append(audio)
            encoded_audio = output_audios

        # Encode Text
        attention_mask = attention_mask.view(1, -1) #1 batch size
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=torch.bfloat16)
        attention_mask = (1.0 - attention_mask) * torch.finfo(torch.bfloat16).min

        hidden_states = self.wte(input_ids.to(self.device))

        kv_seq_len = hidden_states.size()[1]
        ntk_alpha_list = []
        ntk_alpha = self.get_ntk_alpha(kv_seq_len)
        ntk_alpha_list.append(ntk_alpha)
        self.rotary_emb._ntk_alpha_cached_list = ntk_alpha_list
        rotary_pos_emb_list = [
            self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha) for ntk_alpha in ntk_alpha_list
        ]

        head_mask = [None]*32
        past_key_values = (None,)*32
        hidden_states = self.drop(hidden_states)
        encoder_hidden_states = None
        encoder_attention_mask = None
        use_cache = True

        # Unify audio and text
        for idx, (i, a, b) in enumerate(audio_pos):
            hidden_states[i][a : b+1] = encoded_audio[idx]

        # LLM
        presents = () if use_cache else None
        all_self_attentions = None
        all_hidden_states = None
        use_cache = True
        output_attentions = False
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        output_shape = input_shape + (hidden_states.size(-1),)

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                rotary_pos_emb_list=rotary_pos_emb_list,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)


        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        transformer_outputs = BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=presents,
                    hidden_states=all_hidden_states,
                    attentions=all_self_attentions,
                )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        outputs = CausalLMOutputWithPast(
                    loss=loss,
                    logits=lm_logits,
                    past_key_values=transformer_outputs.past_key_values,
                    hidden_states=transformer_outputs.hidden_states,
                    attentions=transformer_outputs.attentions,
                )

        return outputs
