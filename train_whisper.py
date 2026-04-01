import argparse
import os
import re
import numpy as np
import yaml
from datasets import load_dataset, Audio
from num2words import num2words
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from torch.utils.data import IterableDataset as TorchIterableDataset, DataLoader
import torch
import wandb
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import get_peft_model, LoraConfig

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA devices found. Check your NVIDIA driver and PyTorch installation.")
n_gpus = torch.cuda.device_count()
print(f"Found {n_gpus} CUDA device(s):")
for i in range(n_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {props.name}  ({props.total_memory // 1024**2} MiB)")


def get_processor(model_name: str = "openai/whisper-small") -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(model_name, language="danish", task="transcribe")


chars_to_remove_regex = r'[,?\.!\-;:""„%\'\"\»\«]'

def safe_num2words(m):
    try:
        return num2words(int(m.group()), lang='da')
    except Exception:
        return ''

def clean_danish_text(batch):
    text = batch["text"].lower()
    text = re.sub(chars_to_remove_regex, '', text)
    text = re.sub(r'\d+', safe_num2words, text)
    text = re.sub(r"[^a-zæøå ]", "", text)
    text = re.sub(r" +", " ", text).strip()
    batch["text"] = text
    return batch

def is_valid_sample(batch):
    return len(batch["text"].strip()) > 0


def apply_augmentation(array: np.ndarray, sample_rate: int, cfg: dict) -> np.ndarray:
    aug_cfg = cfg.get("augmentation", {})

    if aug_cfg.get("speed_perturbation", False):
        import librosa
        factor = np.random.uniform(0.9, 1.1)
        array = librosa.effects.time_stretch(array.astype(np.float32), rate=factor)

    if aug_cfg.get("noise_augmentation", False):
        noise_level = aug_cfg.get("noise_level", 0.005)
        noise = np.random.randn(len(array)).astype(np.float32) * noise_level
        array = array + noise

    return array


def prepare_dataset(batch, processor, data_cfg):
    audio = batch["audio"]
    array = audio["array"].astype(np.float32)
    sample_rate = audio["sampling_rate"]

    if np.max(np.abs(array)) < 1e-6:
        batch["input_features"] = None
        batch["labels"] = None
        return batch

    min_len = data_cfg.get("min_input_length", 0.5)
    max_len = data_cfg.get("max_input_length", 30.0)
    duration = len(array) / sample_rate
    if duration < min_len or duration > max_len:
        batch["input_features"] = None
        batch["labels"] = None
        return batch

    array = apply_augmentation(array, sample_rate, data_cfg)

    input_features = processor.feature_extractor(array, sampling_rate=16_000).input_features[0]

    if np.any(np.isnan(input_features)):
        batch["input_features"] = None
        batch["labels"] = None
        return batch

    batch["input_features"] = input_features
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

def is_valid_feature(batch):
    return (
        batch["input_features"] is not None
        and batch["labels"] is not None
        and len(batch["labels"]) > 0
    )


# --- HF to TORCH WRAPPER ---
class HFIterableWrapper(TorchIterableDataset):
    def __init__(self, hf_dataset):
        super().__init__()
        self._hf = hf_dataset

    def __iter__(self):
        return iter(self._hf)


def get_dataset(processor, local_rank: int, world_size: int, batch_size: int, data_cfg: dict):

    def _pipeline(split, shuffle=False, augment=False):
        ds = load_dataset(
            "CoRal-project/coral-v2", "read_aloud",
            split=split, streaming=True, trust_remote_code=True,
        )
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
        if shuffle:
            ds = ds.shuffle(
                buffer_size=data_cfg.get("shuffle_buffer", 1_000),
                seed=data_cfg.get("seed", 42),
            )
        ds = ds.map(clean_danish_text)
        ds = ds.filter(is_valid_sample)

        _data_cfg = data_cfg if augment else {**data_cfg, "augmentation": {}}
        ds = ds.map(
            prepare_dataset,
            fn_kwargs={"processor": processor, "data_cfg": _data_cfg},
            remove_columns=["audio", "text"],
        )
        ds = ds.filter(is_valid_feature)
        return IterableDatasetShard(
            HFIterableWrapper(ds),
            batch_size=batch_size,
            drop_last=True,
            num_processes=world_size,
            process_index=local_rank,
        )

    return (
        _pipeline("train", shuffle=True,  augment=True),
        _pipeline("val",   shuffle=False, augment=False),
        _pipeline("test",  shuffle=False, augment=False),
    )


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class WhisperTrainer(Seq2SeqTrainer):
    def _make_dataloader(self, dataset, batch_size: int, num_workers: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            num_workers=num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
            prefetch_factor=(
                self.args.dataloader_prefetch_factor
                if num_workers > 0
                else None
            ),
        )

    def get_train_dataloader(self) -> DataLoader:
        return self._make_dataloader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        n_shards = getattr(dataset, "num_shards", None) or getattr(
            getattr(dataset, "dataset", None), "n_shards", None
        ) or self.args.dataloader_num_workers
        num_workers = min(self.args.dataloader_num_workers, n_shards)
        return self._make_dataloader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            num_workers=num_workers,
        )


class WandbWhisperCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_world_process_zero:
            metrics = {("train_loss" if k == "loss" else k): v
                       for k, v in logs.items() if v is not None}
            metrics["step"] = state.global_step
            wandb.log(metrics)


class UnfreezeEncoderCallback(TrainerCallback):
    def __init__(self, unfreeze_at_step: int):
        self.unfreeze_at_step = unfreeze_at_step
        self._unfrozen = False

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if not self._unfrozen and state.global_step >= self.unfreeze_at_step:
            encoder = model.model.encoder

            for param in encoder.parameters():
                param.requires_grad = True
            self._unfrozen = True

            if state.is_world_process_zero:
                n_params = sum(p.numel() for p in encoder.parameters())
                print(f"\n[Step {state.global_step}] Encoder unfrozen — {n_params:,} params now trainable.")
                wandb.log({"encoder_unfrozen": 1}, step=state.global_step)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="base-whisper.yml")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint dir, or 'true' to auto-resume latest")

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    t_cfg = cfg["training"]

    MODEL_NAME     = m_cfg["name"]
    LANGUAGE       = m_cfg["language"]
    TASK           = m_cfg["task"]
    FREEZE_ENCODER = m_cfg.get("freeze_encoder", True)
    USE_LORA       = m_cfg.get("use_lora", True)
    REPO_NAME      = t_cfg["output_dir"]

    unfreeze_step = m_cfg.get("unfreeze_encoder_at_step", None)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if local_rank == 0:
        wandb.init(project="asr-danish-whisper", entity="abargum", config=cfg)

    processor  = get_processor(MODEL_NAME)
    wer_metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        pred_norm  = [normalizer(s) for s in pred_str]
        label_norm = [normalizer(s) for s in label_str]

        pairs = [(p, l) for p, l in zip(pred_norm, label_norm) if len(l) > 0]
        if not pairs:
            return {"wer": 1.0}
        pred_norm, label_norm = zip(*pairs)

        wer = wer_metric.compute(predictions=list(pred_norm), references=list(label_norm))
        return {"wer": wer}

    train_data, val_data, _ = get_dataset(
        processor,
        local_rank=local_rank,
        world_size=world_size,
        batch_size=t_cfg["batch_size"],
        data_cfg=d_cfg,
    )
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    attention_dropout = m_cfg.get("attention_dropout", 0.0)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        attention_dropout=attention_dropout,
    )

    if FREEZE_ENCODER:
        model.freeze_encoder()

    if USE_LORA:
        lora_config = LoraConfig(
            r=m_cfg.get("lora_r", 8),
            lora_alpha=m_cfg.get("lora_alpha", 16),
            target_modules=m_cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
            lora_dropout=m_cfg.get("lora_dropout", 0.05),
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    training_args = Seq2SeqTrainingArguments(
        output_dir=REPO_NAME,
        per_device_train_batch_size=t_cfg["batch_size"],
        gradient_accumulation_steps=t_cfg["grad_accum_steps"],
        evaluation_strategy="steps",
        per_device_eval_batch_size=t_cfg["eval_batch_size"],
        predict_with_generate=True,
        generation_max_length=t_cfg["generation_max_length"],
        num_train_epochs=10,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,
        bf16=False,
        save_steps=t_cfg["save_steps"],
        eval_steps=t_cfg["eval_steps"],
        logging_steps=t_cfg["logging_steps"],
        learning_rate=t_cfg["learning_rate"],
        warmup_steps=t_cfg["warmup_steps"],
        save_total_limit=2,
        load_best_model_at_end=t_cfg["load_best_model_at_end"],
        metric_for_best_model=t_cfg["metric_for_best_model"],
        greater_is_better=t_cfg["greater_is_better"],
        push_to_hub=False,
        max_steps=t_cfg["max_steps"],
        dataloader_num_workers=t_cfg["dataloader_num_workers"],
        dataloader_prefetch_factor=2 if t_cfg["dataloader_num_workers"] > 0 else None,
        adam_beta2=0.98,
        ddp_find_unused_parameters=False,
        ignore_data_skip=True,
        remove_unused_columns=False,
    )

    callbacks = [WandbWhisperCallback()]
    if FREEZE_ENCODER and unfreeze_step is not None:
        callbacks.append(UnfreezeEncoderCallback(unfreeze_at_step=unfreeze_step))
    
    trainer = WhisperTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,
        callbacks=callbacks,          
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    processor.save_pretrained(REPO_NAME)