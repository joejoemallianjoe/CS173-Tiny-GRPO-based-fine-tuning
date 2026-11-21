# ----------------------------------------------------------------------
# Minimal GRPO fine‑tuning for a vowel‑count task using HuggingFace TRL
# ----------------------------------------------------------------------
# This implementation follows the GRPO objective described in DeepSeek’s
# original paper (see DeepSeekMath corpus description for training setup) [1].
# ----------------------------------------------------------------------

import os
from typing import List, Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import GRPOTrainer
from trl.trainer import TrainerCallback

# ----------------------------------------------------------------------
# 1. Load the CSV dataset (expects columns: "prompt" and "answer")
# ----------------------------------------------------------------------
# Example CSV rows:
# prompt,answer
# "Count vowels in: hello world",5
# "Count vowels in: AI",2
# ----------------------------------------------------------------------
dataset = load_dataset("csv", data_files="data.csv")["train"]

# ----------------------------------------------------------------------
# 2. Tokenizer & model (replace with any suitable instruction‑tuned LLM)
# ----------------------------------------------------------------------
model_name = "Qwen/Qwen2.5-0.5B-Instruct"          # can be swapped for a local checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

# ----------------------------------------------------------------------
# 3. Reward function – returns 1.0 if the model’s vowel count matches the
#    reference answer, otherwise 0.0.
# ----------------------------------------------------------------------
VOWELS = set("aeiouAEIOU")

def vowel_count_reward(prompts: List[str], responses: List[str], references: List[int]) -> List[float]:
    """Calculate binary reward for each (prompt, response, reference) triple."""
    rewards = []
    for resp, ref in zip(responses, references):
        # count vowels in the model’s raw text response
        pred_cnt = sum(1 for ch in resp if ch in VOWELS)
        rewards.append(1.0 if pred_cnt == ref else 0.0)
    return rewards

# Wrapper required by GRPOTrainer: it receives a list of dicts with
# "prompt", "response", and "reference" fields.
def reward_wrapper(samples: List[Dict]) -> List[float]:
    prompts   = [s["prompt"]   for s in samples]
    # the model will generate the responses later; here we just pass placeholders
    # GRPOTrainer will replace them with the actual generated text.
    responses = [s.get("response", "") for s in samples]
    references = [s["answer"] for s in samples]
    return vowel_count_reward(prompts, responses, references)

# ----------------------------------------------------------------------
# 4. Logging callback – records reward per training step and plots at the end
# ----------------------------------------------------------------------
class RewardLoggingCallback(TrainerCallback):
    def __init__(self):
        self.rewards = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # GRPOTrainer logs a key named "reward" after each update step
        if logs is not None and "reward" in logs:
            self.rewards.append(logs["reward"])

    def on_train_end(self, args, state, control, **kwargs):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(self.rewards, label="Reward (binary)")
        plt.xlabel("Training steps")
        plt.ylabel("Reward")
        plt.title("GRPO reward over training")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ----------------------------------------------------------------------
# 5. Training arguments (industry‑standard defaults)
# ----------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./grpo_vowel",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    fp16=True,                     # use mixed precision if GPU supports it
    report_to=[],                  # disable external logging (wandb, etc.) for this minimal example
)

# ----------------------------------------------------------------------
# 6. Instantiate GRPOTrainer
# ----------------------------------------------------------------------
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_wrapper,          # our binary reward function
    callbacks=[RewardLoggingCallback()],  # visual reward tracking
)

# ----------------------------------------------------------------------
# 7. Start training
# ----------------------------------------------------------------------
trainer.train()
