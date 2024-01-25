from CustomQWen.generation_utils_sample import CustomGenerationMixin
from transformers import AutoTokenizer
import torch
from CustomQWen.config_qwen import QWenConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from types import SimpleNamespace

# Config
torch.manual_seed(0)
device = "cuda"

qwen_config = QWenConfig()
with open('/home/jbellver/repos/Multimodal-Audio-Recognition/QWenFinetune/config.json', 'r') as file:
    config = json.load(file)
with open('/home/jbellver/repos/Multimodal-Audio-Recognition/QWenFinetune/generation_config.json', 'r') as file:
    generation_config = json.load(file)
config = SimpleNamespace(**config)
generation_config = SimpleNamespace(**generation_config)

# Load modules
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio", trust_remote_code=True, padding=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio", trust_remote_code=True).to("cpu")

generate = CustomGenerationMixin(model,
                                 device,
                                 config,
                                 qwen_config,
                                 generation_config
                                )

# Sample
audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1272-128104-0000.flac"
sp_prompt = "<|startoftranscription|><|en|><|transcribe|><|en|><|notimestamps|><|wo_itn|>"
query = f"<audio>{audio_url}</audio>{sp_prompt}"
audio_info = tokenizer.process_audio(query)
inputs = tokenizer(query, return_tensors='pt', audio_info=audio_info)
inputs = inputs.to(device)

output = generate.sample(
        audio_info=audio_info,
        input_ids = inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        token_type_ids=inputs['token_type_ids'],
        logits_processor = None,
        stopping_criteria = None,
        logits_warper = None,
        max_length = None,
        pad_token_id = 151643,
        eos_token_id = 151643,
        output_attentions = False,
        output_hidden_states = False,
        output_scores = False,
        return_dict_in_generate = False,
        synced_gpus = False,
        streamer = None
    )

response = tokenizer.decode(output.cpu()[0], skip_special_tokens=False,audio_info=audio_info)
print(response)
