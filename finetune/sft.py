import argparse
import logging

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a software engineer.
Answer the provided question using only the provided code and documentation.
Do not invent things that are not present.
Be technical, precise, and concise.
Do not restate the question."""


def build_user_prompt(code: str, documentation: str, question: str) -> str:
    return f"""Code:
{code}

Documentation:
{documentation}

Question:
{question}

Answer directly and technically."""


def to_prompt_completion(example):
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                (example.get("code") or "").strip(),
                (example.get("text") or "").strip(),
                (example.get("question") or "").strip(),
            ),
        },
    ]

    completion_messages = [
        {
            "role": "assistant",
            "content": (example.get("improved_answer") or "").strip(),
        }
    ]

    return {
        "prompt": prompt_messages,
        "completion": completion_messages,
    }


def keep_example(x):
    return (
        bool(x.get("code"))
        and bool(x.get("text"))
        and bool(x.get("question"))
        and bool(x.get("improved_answer"))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--model_name",
        type=str,
        default="/data/rech/jaouaime/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--max_seq_length", type=int, default=8000)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=4.0)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=20)

    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--packing", action="store_true", default=False)
    parser.add_argument("--merge_weights", action="store_true", default=False)

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Stop if eval loss does not improve for this many evals. 0 to disable.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available. Training will run on CPU and be very slow."
        )

    # Load train dataset
    train_ds = load_dataset("json", data_files=args.train_file)["train"]
    train_ds = train_ds.filter(keep_example)
    logger.info(f"Train size after filtering: {len(train_ds)}")

    # Load eval dataset if provided
    if args.eval_file is not None:
        eval_ds = load_dataset("json", data_files=args.eval_file)["train"]
        eval_ds = eval_ds.filter(keep_example)
        logger.info(f"Eval size after filtering: {len(eval_ds)}")
    else:
        eval_ds = None
        logger.info("No eval file provided — early stopping disabled.")

    # Convert to prompt-completion format
    train_remove_cols = train_ds.column_names
    train_ds = train_ds.map(to_prompt_completion, remove_columns=train_remove_cols)

    if eval_ds is not None:
        eval_remove_cols = eval_ds.column_names
        eval_ds = eval_ds.map(to_prompt_completion, remove_columns=eval_remove_cols)

    # QLoRA 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    

    use_bf16 = torch.cuda.is_available()
    use_early_stopping = eval_ds is not None and args.early_stopping_patience > 0

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_ds is not None else None,
        eval_strategy="steps" if eval_ds is not None else "no",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model="eval_loss" if use_early_stopping else None,
        greater_is_better=False if use_early_stopping else None,
        bf16=use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_seq_length,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.0,
        report_to="none",
        packing=args.packing,
        completion_only_loss=True,
    )

    callbacks = None
    if use_early_stopping:
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ]
        logger.info(
            f"Early stopping enabled with patience={args.early_stopping_patience}."
        )
    else:
        logger.info("Early stopping disabled (no eval file or patience=0).")

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"LoRA adapter saved to {args.output_dir}")

    if args.merge_weights:
        logger.info("Merging LoRA weights into base model...")
        merged_dir = args.output_dir + "/merged"
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        logger.info(f"Merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()