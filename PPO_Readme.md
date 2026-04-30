# 3. Fine-Tuning with PPO

After preparing the model, the next step is to fine-tune the student model using the **PPO (Proximal Policy Optimization)** algorithm.

---

## Setup

First, install the required dependencies listed in `ppo_requirements.txt`:

```bash
pip install -r ppo_requirements.txt
```

---

## Project Structure

The main PPO script is located in:

```text
finetune/
```

### Main Script

```text
finetune/ppo.py
```

---

## Running the Training Phases

The PPO pipeline is executed in **three sequential phases**. Each phase must be run separately.

---

### 1. SFT Phase (Supervised Fine-Tuning)

Run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python ppo.py --phase sft
```

---

### 2. Teacher Cache Phase

Run the caching step:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python ppo.py --phase cache
```

---

### 3. PPO Training Phase

Finally, run the PPO optimization:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python ppo.py --phase ppo
```
