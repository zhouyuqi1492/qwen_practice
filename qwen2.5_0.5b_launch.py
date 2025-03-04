import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Now you do not need to add "trust_remote_code=True"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct",
                                             torch_dtype="auto",
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

# Instead of using model.chat(), we directly use model.generate()
# But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
prompt = "Give me a short introduction to large language model."
messages = [{
    "role": "system",
    "content": "You are a helpful assistant."
}, {
    "role": "user",
    "content": prompt
}]
text = tokenizer.apply_chat_template(messages,
                                     tokenize=False,
                                     add_generation_prompt=True)
# print(text)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# Directly use generate() and tokenizer.decode() to get the output.
# Use `max_new_tokens` to control the maximum output length.
generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
# print('model_inputs.input_ids ', model_inputs.input_ids.shape)
# print('generated_ids', generated_ids.shape)
generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# print(response)

# Generate output by streaming.
streamer = TextIteratorStreamer(tokenizer,
                                skip_prompt=True,
                                skip_special_tokens=True)

generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
thread = Thread(target=model.generate, kwargs=generation_kwargs)

thread.start()
generated_text = ""
for new_text in streamer:
    # time.sleep(0.5)
    # print('add:', new_text)
    generated_text += new_text
print(generated_text)
