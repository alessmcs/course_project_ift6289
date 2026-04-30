import argparse
import inspect
import logging

import torch
from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed,
)
from trl import DPOConfig, DPOTrainer

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


def to_preference(example, chosen_key="improved_answer", rejected_key="raw_answer", doc_key="text"):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    (example.get("code") or "").strip(),
                    (example.get(doc_key) or "").strip(),
                    (example.get("question") or "").strip(),
                ),
            },
        ],
        "chosen":   [{"role": "assistant", "content": (example.get(chosen_key)   or "").strip()}],
        "rejected": [{"role": "assistant", "content": (example.get(rejected_key) or "").strip()}],
    }


def keep_example(example, chosen_key="improved_answer", rejected_key="raw_answer", doc_key="text"):
    chosen   = (example.get(chosen_key)   or "").strip()
    rejected = (example.get(rejected_key) or "").strip()
    return (
        bool(example.get("code"))
        and bool(example.get(doc_key))
        and bool(example.get("question"))
        and bool(chosen)
        and bool(rejected)
        and chosen != rejected
    )


def supports(cls, name: str) -> bool:
    """Check if a class supports a parameter, walking the full MRO for dataclasses."""
    try:
        if name in inspect.signature(cls.__init__).parameters:
            return True
        for base in inspect.getmro(cls):
            if name in base.__dict__:
                return True
            try:
                if name in inspect.signature(base.__init__).parameters:
                    return True
            except Exception:
                pass
        return False
    except Exception:
        return False


def load_bnb_base(model_name: str, bnb_config: BitsAndBytesConfig) -> AutoModelForCausalLM:
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    # use_gradient_checkpointing=True keeps PEFT consistent with gradient_checkpointing=True
    # in DPOConfig — PEFT sets up the input-embedding hooks needed for 4-bit training.
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    return base


def _verify_disable_adapter(model: PeftModel, tokenizer: AutoTokenizer) -> None:
    """
    Verify that disable_adapter() produces different logits from the adapted forward pass.
    If outputs are identical the SFT adapter is not active or disable_adapter() is broken,
    meaning TRL will compute wrong reference log-probs silently.
    """
    dummy = tokenizer("hello world", return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        with model.disable_adapter():
            ref_logits = model(**dummy).logits
        ada_logits = model(**dummy).logits
    model.train()

    if torch.allclose(ref_logits, ada_logits, atol=1e-4):
        logger.warning(
            "SANITY CHECK FAILED: disable_adapter() output is identical to adapted output. "
            "The SFT adapter may not be active or disable_adapter() is broken. "
            "Reference log-probs will be INCORRECT. Training will proceed but results "
            "may be invalid — inspect your SFT checkpoint path."
        )
    else:
        logger.info("SANITY CHECK PASSED: disable_adapter() produces different logits as expected.")


def build_ref_model(model: PeftModel, tokenizer: AutoTokenizer) -> PeftModel | None:
    """
    Build a reference model WITHOUT loading a second copy of base weights on GPU.

    Strategy 1 — trl.trainer.utils.create_reference_model (TRL >= 0.12):
        Creates a CPU-offloaded or frozen deepcopy of the model. Costs no extra
        GPU VRAM for the base weights. Preferred.

    Strategy 2 — ref_model=None:
        TRL calls model.disable_adapter() internally on each ref forward pass.
        Verified with a sanity check to catch silent failures. No extra VRAM at all.

    An explicit second GPU model load is intentionally omitted: on a 24 GB GPU
    with an 8B model already loaded in 4-bit (~19 GB), a second load causes OOM.
    """

    # ── Strategy 1: TRL's create_reference_model ──────────────────────────────
    try:
        from trl.trainer.utils import create_reference_model
        ref_model = create_reference_model(model)
        logger.info("Reference model built via trl.trainer.utils.create_reference_model.")
        return ref_model
    except (ImportError, AttributeError) as e:
        logger.info(
            f"create_reference_model not available ({e}). "
            "Falling back to ref_model=None + disable_adapter()."
        )

    # ── Strategy 2: None + disable_adapter() sanity check ─────────────────────
    logger.warning(
        "Using ref_model=None. TRL will use disable_adapter() for reference log-probs. "
        "Running sanity check..."
    )
    _verify_disable_adapter(model, tokenizer)
    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file",       type=str, required=True)
    parser.add_argument("--eval_file",         type=str, default=None)
    parser.add_argument("--output_dir",        type=str, required=True)
    parser.add_argument("--base_model_name",   type=str,
                        default="/data/rech/jaouaime/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--sft_checkpoint",    type=str, required=True,
                        help="Path to the SFT LoRA adapter checkpoint.")

    parser.add_argument("--doc_key",           type=str, default="text")
    parser.add_argument("--chosen_key",        type=str, default="improved_answer")
    parser.add_argument("--rejected_key",      type=str, default="raw_answer")

    parser.add_argument("--max_seq_length",        type=int,   default=8000)
    parser.add_argument("--max_prompt_length",     type=int,   default=6000)
    parser.add_argument("--max_completion_length", type=int,   default=512)
    parser.add_argument("--learning_rate",         type=float, default=1e-6)
    parser.add_argument("--beta",                  type=float, default=0.1)
    parser.add_argument("--num_train_epochs",      type=float, default=2.0)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size",  type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--save_steps",    type=int, default=100)
    parser.add_argument("--eval_steps",    type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--warmup_steps",  type=int, default=30)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    args = parser.parse_args()
    set_seed(args.seed)

    # ── Dataset ────────────────────────────────────────────────────────────────

    fn_kwargs = {
        "chosen_key":   args.chosen_key,
        "rejected_key": args.rejected_key,
        "doc_key":      args.doc_key,
    }

    train_ds = load_dataset("json", data_files=args.train_file)["train"]
    train_ds = train_ds.filter(keep_example, fn_kwargs=fn_kwargs)
    train_ds = train_ds.map(to_preference, fn_kwargs=fn_kwargs, remove_columns=train_ds.column_names)
    logger.info(f"Train size: {len(train_ds)}")

    eval_ds = None
    if args.eval_file:
        eval_ds = load_dataset("json", data_files=args.eval_file)["train"]
        eval_ds = eval_ds.filter(keep_example, fn_kwargs=fn_kwargs)
        eval_ds = eval_ds.map(to_preference, fn_kwargs=fn_kwargs, remove_columns=eval_ds.column_names)
        logger.info(f"Eval size: {len(eval_ds)}")

    # ── Tokenizer ──────────────────────────────────────────────────────────────

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for DPO

    # ── BnB config ─────────────────────────────────────────────────────────────

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # ── Model ──────────────────────────────────────────────────────────────────

    logger.info("Loading base model + SFT adapter (trainable).")
    base = load_bnb_base(args.base_model_name, bnb_config)
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.eos_token_id = tokenizer.eos_token_id
    base.config.use_cache    = False

    model = PeftModel.from_pretrained(
        base,
        args.sft_checkpoint,
        is_trainable=True,
    )

    # Align model config + generation config with tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        if tokenizer.bos_token_id is not None:
            model.generation_config.bos_token_id = tokenizer.bos_token_id

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    # ── Reference model ────────────────────────────────────────────────────────
    # No second GPU model load — see build_ref_model() for full rationale.

    ref_model = build_ref_model(model=model, tokenizer=tokenizer)

    # precompute_ref_log_probs=True is only useful when ref_model=None
    # (the disable_adapter path). With an explicit ref_model TRL calls it
    # directly each step; precomputing would waste memory storing the full
    # dataset's log-probs.
    precompute = ref_model is None

    # ── DPO config ─────────────────────────────────────────────────────────────

    use_early_stopping = eval_ds is not None and args.early_stopping_patience > 0

    dpo_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        eval_steps=args.eval_steps if eval_ds else None,
        eval_strategy="steps" if eval_ds else "no",
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model="eval_loss" if use_early_stopping else None,
        greater_is_better=False if use_early_stopping else None,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.0,
        report_to="none",
        precompute_ref_log_probs=precompute,
    )

    # Length caps — check both DPOConfig and fall through to trainer if needed
    for key, val in [
        ("max_length",            args.max_seq_length),
        ("max_prompt_length",     args.max_prompt_length),
        ("max_completion_length", args.max_completion_length),
    ]:
        if supports(DPOConfig, key):
            dpo_kwargs[key] = val

    dpo_args = DPOConfig(**dpo_kwargs)

    # ── Trainer ────────────────────────────────────────────────────────────────

    trainer_kwargs = dict(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[EarlyStoppingCallback(args.early_stopping_patience)] if use_early_stopping else None,
    )

    if supports(DPOTrainer, "processing_class"):
        trainer_kwargs["processing_class"] = tokenizer
    elif supports(DPOTrainer, "tokenizer"):
        trainer_kwargs["tokenizer"] = tokenizer

    # Fallback: pass length args to trainer if DPOConfig doesn't support them
    for key, val in [
        ("max_length",            args.max_seq_length),
        ("max_prompt_length",     args.max_prompt_length),
        ("max_completion_length", args.max_completion_length),
    ]:
        if not supports(DPOConfig, key) and supports(DPOTrainer, key):
            trainer_kwargs[key] = val

    trainer = DPOTrainer(**trainer_kwargs)

    # ── Train ──────────────────────────────────────────────────────────────────

    logger.info("Starting DPO training...")
    trainer.train()
    logger.info("DPO training complete.")

    # Save the trained adapter
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"DPO adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()