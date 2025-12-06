"""
简化测试：演示为什么必须先 prepare 再 load
"""

import os
import shutil
import tempfile

import torch
from accelerate import Accelerator

print("=" * 70)
print("为什么必须先 prepare 再 load？")
print("=" * 70)

# 创建临时目录
temp_dir = tempfile.mkdtemp()
save_dir = os.path.join(temp_dir, "checkpoint")

# ============================================================================
# 步骤1: 训练并保存模型（使用 accelerator.save_state）
# ============================================================================
print("\n【步骤1】训练并保存模型")
print("-" * 70)

model = torch.nn.Linear(10, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 设置特定的权重值（模拟训练后的权重）
with torch.no_grad():
    model.weight.fill_(5.0)
    model.bias.fill_(10.0)

accelerator = Accelerator()
model, optimizer = accelerator.prepare(model, optimizer)

print(f"训练后的权重和: {model.weight.sum().item():.1f}")
print(f"训练后的偏置和: {model.bias.sum().item():.1f}")

# 使用 accelerator.save_state 保存
accelerator.save_state(save_dir)
print(f"✅ 已保存到: {save_dir}")

# ============================================================================
# 步骤2: 错误方式 - 先 load 再 prepare
# ============================================================================
print("\n【步骤2】❌ 错误方式：先 load 再 prepare")
print("-" * 70)

model_wrong = torch.nn.Linear(10, 10)
optimizer_wrong = torch.optim.SGD(model_wrong.parameters(), lr=0.01)

# 初始化为0（模拟新模型）
with torch.no_grad():
    model_wrong.weight.fill_(0.0)
    model_wrong.bias.fill_(0.0)

print(f"初始权重和: {model_wrong.weight.sum().item():.1f}")

accelerator_wrong = Accelerator()

# 尝试在 prepare 之前 load
model_wrong, optimizer_wrong = accelerator_wrong.prepare(model_wrong, optimizer_wrong)
accelerator_wrong.load_state(save_dir, load_kwargs={"weights_only": False})

print(f"load 后权重和: {model_wrong.weight.sum().item():.1f} ✅")
print(f"load 后偏置和: {model_wrong.bias.sum().item():.1f} ✅")

exit(0)

# ============================================================================
# 步骤3: 正确方式 - 先 prepare 再 load
# ============================================================================
print("\n【步骤3】✅ 正确方式：先 prepare 再 load")
print("-" * 70)

model_correct = torch.nn.Linear(10, 10)
optimizer_correct = torch.optim.SGD(model_correct.parameters(), lr=0.01)

# 初始化为0（模拟新模型）
with torch.no_grad():
    model_correct.weight.fill_(0.0)
    model_correct.bias.fill_(0.0)

print(f"初始权重和: {model_correct.weight.sum().item():.1f}")
print(f"初始偏置和: {model_correct.bias.sum().item():.1f}")

accelerator_correct = Accelerator()

# 先 prepare
model_correct, optimizer_correct = accelerator_correct.prepare(model_correct, optimizer_correct)
print(f"\nprepare 后权重和: {model_correct.weight.sum().item():.1f} (仍然是0)")

# 再 load
accelerator_correct.load_state(save_dir, load_kwargs={"weights_only": False})
print(f"load 后权重和: {model_correct.weight.sum().item():.1f} ✅")
print(f"load 后偏置和: {model_correct.bias.sum().item():.1f} ✅")

# 清理
shutil.rmtree(temp_dir)

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 70)
print("总结")
print("=" * 70)
print(
    """
使用 accelerator.save_state() 和 load_state() 时：

1. save_state() 保存的是 PREPARED 模型的状态
2. load_state() 必须加载到 PREPARED 模型

正确顺序：
  model, optimizer = accelerator.prepare(model, optimizer)  # 先 prepare
  accelerator.load_state(checkpoint_dir)                     # 再 load

错误顺序：
  accelerator.load_state(checkpoint_dir)                     # ❌ 会失败
  model, optimizer = accelerator.prepare(model, optimizer)  
"""
)
