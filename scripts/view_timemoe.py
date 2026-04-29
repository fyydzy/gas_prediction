import inspect
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Maple728/TimeMoE-50M",
    trust_remote_code=True,
)

# print("forward signature:", inspect.signature(model.forward))

# 看关键线性层名字（用于 LoRA target_modules）
for n, m in model.named_modules():
    if m.__class__.__name__ == "Linear":
        if any(k in n.lower() for k in ["q", "k", "v", "o", "gate", "up", "down", "proj", "expert"]):
            print(n)