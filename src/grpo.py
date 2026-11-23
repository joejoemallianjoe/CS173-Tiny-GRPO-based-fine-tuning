#Minimal GRPO fine-tuning for a vowel-count task using HuggingFace TRL
# ----------------------------------------------------------------------
# Hyperparameters are exactly those recommended by the original LLM output.
# Corrections applied after the code audit (marked "FIX" below):
#   1. Reward reads the integer the model STATES (was: counted vowels in the reply)
#   2. Reward signature matches TRL: (prompts, completions, **dataset_columns)
#   3. GRPOConfig instead of TrainingArguments
#   4. processing_class= instead of tokenizer=
#   5. TrainerCallback imported from transformers (not trl.trainer)
#   6. bf16=True to match the bfloat16 model (was fp16=True)
#   7. Reward curve saved to reward_curve.png (plt.show() is lost in headless runs)
# Dataset: data.csv with columns "prompt" and "answer" (true vowel count).
# ----------------------------------------------------------------------

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback  # FIX: import location
from trl import GRPOConfig, GRPOTrainer                                        # FIX: GRPOConfig

# 1. Dataset (CSV with columns "prompt" and "answer")
dataset = load_dataset("csv", data_files="data.csv")["train"]
# FIX: tell the model to answer with just a number so the reward can parse it
dataset = dataset.map(lambda ex: {"prompt": "Count the vowels in: " + ex["prompt"] + "\nAnswer with only a single integer."})

# 2. Tokenizer & model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)


# 3. Reward: 1.0 if the integer the model STATES equals the true vowel count
# FIX: original counted vowels in the response text instead of reading the stated number;
#      signature now matches TRL: (prompts, completions, **dataset_columns)
def vowel_count_reward(prompts, completions, answer, **kwargs):
    rewards = []
    for completion, ref in zip(completions, answer):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        nums = re.findall(r"-?\d+", text)
        pred = int(nums[-1]) if nums else None
        rewards.append(1.0 if pred == int(ref) else 0.0)
    return rewards


# 4. Log reward over time and plot at the end
class RewardLoggingCallback(TrainerCallback):
    def __init__(self):
        self.steps, self.rewards = [], []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "reward" in logs:
            self.steps.append(state.global_step)
            self.rewards.append(logs["reward"])

    def on_train_end(self, args, state, control, **kwargs):
        plt.figure(figsize=(8, 4))
        plt.plot(self.steps, self.rewards, label="Reward (binary)")
        plt.xlabel("Training steps")
        plt.ylabel("Reward")
        plt.title("GRPO reward over training")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("reward_curve.png", dpi=150)  # FIX: save instead of show()


# 5. Training arguments (LLM's original values)
training_args = GRPOConfig(                      # FIX: GRPOConfig instead of TrainingArguments
    output_dir="./grpo_vowel",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=1,
    logging_steps=1,
    save_steps=200,
    bf16=True,                                   # FIX: fp16=True conflicted with the bfloat16 model
    report_to=[],
)

# 6. Trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,                  # FIX: was tokenizer=
    args=training_args,
    train_dataset=dataset,
    reward_funcs=vowel_count_reward,
    callbacks=[RewardLoggingCallback()],
)

# 7. Train
trainer.train()
