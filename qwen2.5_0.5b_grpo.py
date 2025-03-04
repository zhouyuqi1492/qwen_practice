from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from datasets import load_dataset, Dataset
from modelscope.msdatasets import MsDataset
from trl import GRPOConfig, GRPOTrainer
import argparse
import os

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    match = re.search('<answer>(.*)</answer>', text, re.DOTALL)
    if match:
        answer = match.group(1)
    else:
        answer = ''
    return answer.strip()


def correctness_reward_func(prompts, completions, answer,
                            **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-' * 20, f"Question:\n{q}", f"\nResponse:\n{responses[0]}",
          f"\nExtracted:\n{extracted_responses[0]}", f"\nAnswer:\n{answer[0]}")
    return [1 if a in r else 0.0 for r, a in zip(extracted_responses, answer)]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [2 if match else 0.0 for match in matches]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [4 if match else 0.0 for match in matches]


# Traning set.
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_questions(split="train") -> Dataset:
    data = MsDataset.load('modelscope/gsm8k',
                          subset_name='main',
                          split='train')
    # print('--------Before---------')
    # print(data[0])
    data = data.map(
        lambda x: {
            'prompt': [{
                'role': 'system',
                'content': SYSTEM_PROMPT
            }, {
                'role': 'user',
                'content': '数字10203040里面有几个0?'
            }, {
                'role':
                'assistant',
                'content':
                XML_COT_FORMAT.format(reasoning=
                                      '可以将数字拆开看，1、0、2、0、3、0、4、0，我们可以数出有4个0',
                                      answer='4')
            }, {
                'role': 'user',
                'content': x['question']
            }],
            'answer':
            extract_hash_answer(x['answer'])
        })
    return data


def train():
    model_name = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = get_gsm8k_questions()
    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=2,
        max_prompt_length=256,
        max_completion_length=300,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        vllm_gpu_memory_utilization=0.2,
        report_to="tensorboard",
        output_dir="outputs/Qwen2.5-1.5B-Instruct-GRPO",
    )

    trainer = GRPOTrainer(
        model=model_name,
        processing_class=tokenizer,
        reward_funcs=[
            soft_format_reward_func,
            strict_format_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()


def infer():
    query = '小明站在队伍中间，前面有2个人，后面有3个人，请问队伍一共多少人？'
    messages = [{
        'role': 'system',
        'content': SYSTEM_PROMPT
    }, {
        'role': 'user',
        'content': '数字10203040里面有几个0?'
    }, {
        'role':
        'assistant',
        'content':
        XML_COT_FORMAT.format(reasoning='可以将数字拆开看，1、0、2、0、3、0、4、0，我们可以数出有4个0',
                              answer='4')
    }, {
        'role': 'user',
        'content': query
    }]

    # encode
    checkpoints = os.listdir('./outputs/Qwen2.5-1.5B-Instruct-GRPO')
    latest_checkpoints = sorted(filter(lambda x: x.startswith('checkpoint'),
                                       checkpoints),
                                key=lambda x: int(x.split('-')[-1]))[-1]

    model_name = f'./outputs/Qwen2.5-1.5B-Instruct-GRPO/{latest_checkpoints}'
    model_name = f'./outputs/Qwen2.5-1.5B-Instruct-GRPO/checkpoint-900'
    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = tokenizer.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    # RL model version
    print('--------RL Model---------')
    grpo_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      torch_dtype="auto",
                                                      device_map="auto")
    generated_ids_rl = grpo_model.generate(
        **model_inputs,
        max_new_tokens=300,
    )
    completion_ids_rl = generated_ids_rl[0][len(model_inputs.input_ids[0]):]
    completion_text_rl = tokenizer.decode(completion_ids_rl,
                                          skip_special_tokens=True)
    print(completion_text_rl)

    # raw model version
    print('--------Raw Model---------')
    raw_model = AutoModelForCausalLM.from_pretrained(
        '/home/yukizh/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct',
        torch_dtype="auto",
        device_map="auto")
    generated_ids_raw = raw_model.generate(
        **model_inputs,
        max_new_tokens=300,
    )
    completion_ids_raw = generated_ids_raw[0][len(model_inputs.input_ids[0]):]
    completion_text_raw = tokenizer.decode(completion_ids_raw,
                                           skip_special_tokens=True)
    print(completion_text_raw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run training or inference based on the provided argument."
    )
    parser.add_argument("--mode", type=str, help="train or infer")

    # print('--------Start Training---------')
    args = parser.parse_args()

    if args.mode == "train":
        print('Start training ... \n')
        train()
    elif args.mode == 'infer':
        print('Start inferenceing ... \n')
        infer()
    else:
        print("Invalid mode. Please use 'train' or 'infer'.")
