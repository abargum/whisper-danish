import argparse
import re
import csv
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, Audio
from num2words import num2words
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate

chars_to_remove_regex = r'[,?\.!\-;:""„%\'\"\»\«]'
_normalizer = BasicTextNormalizer()

def safe_num2words(m):
    try:
        return num2words(int(m.group()), lang="da")
    except Exception:
        return ""

def clean_danish_text(text: str) -> str:
    text = text.lower()
    text = re.sub(chars_to_remove_regex, "", text)
    text = re.sub(r"\d+", safe_num2words, text)
    text = re.sub(r"[^a-zæøå ]", "", text)
    text = re.sub(r" +", " ", text).strip()
    return text

def normalise(text: str) -> str:
    """Light normalisation used for WER/CER scoring."""
    return _normalizer(text).strip()

def compute_cer(predictions, references):
    """Character Error Rate via jiwer on character-split strings."""
    pred_chars = [" ".join(list(p)) for p in predictions]
    ref_chars  = [" ".join(list(r)) for r in references]
    wer_metric = evaluate.load("wer")
    return wer_metric.compute(predictions=pred_chars, references=ref_chars)

def build_pipeline(model_id_or_path: str, device: str, torch_dtype):
    """Generic pipeline builder for any Whisper-compatible model."""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id_or_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id_or_path)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": "danish", "task": "transcribe"},
    )
    return pipe

def build_finetuned_pipeline(checkpoint_path: str, device: str, torch_dtype):
    try:
        return build_pipeline(checkpoint_path, device, torch_dtype)
    except Exception as e:
        print(f"  [warn] AutoModel load failed ({e}), trying WhisperForConditionalGeneration …")
        model = WhisperForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype,
            ignore_mismatched_sizes=True,
        )
        model.to(device)
        try:
            processor = WhisperProcessor.from_pretrained(checkpoint_path)
        except Exception:
            print("  [warn] Processor not found in checkpoint, loading openai/whisper-tiny")
            processor = WhisperProcessor.from_pretrained(
                "openai/whisper-tiny", language="danish", task="transcribe"
            )
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            generate_kwargs={"language": "danish", "task": "transcribe"},
        )
        return pipe

def evaluate_pipeline(pipe, dataset, batch_size: int, label: str):
    """
    Run ASR pipeline over the full dataset and return (wer, cer).

    dataset: HF dataset with "audio" (dict with array+sampling_rate) and "text" columns.
    """
    wer_metric = evaluate.load("wer")

    all_preds  = []
    all_labels = []
    skipped    = 0
    total      = len(dataset)

    print(f"\n{'='*60}")
    print(f"  Evaluating: {label}  ({total} samples, batch={batch_size})")
    print(f"{'='*60}")

    for start in range(0, total, batch_size):
        end   = min(start + batch_size, total)
        batch = dataset.select(range(start, end))

        audios = [s["audio"] for s in batch]
        texts  = [s["text"]  for s in batch]

        valid_audios, valid_texts = [], []
        for audio, text in zip(audios, texts):
            arr = np.array(audio["array"], dtype=np.float32)
            ref = clean_danish_text(text)
            if np.max(np.abs(arr)) < 1e-6 or len(ref) == 0:
                skipped += 1
                continue
            valid_audios.append(audio)
            valid_texts.append(ref)

        if not valid_audios:
            continue

        try:
            results = pipe(valid_audios, batch_size=batch_size)
        except Exception as e:
            print(f"  [error] batch {start}–{end}: {e} — skipping")
            skipped += len(valid_audios)
            continue

        for result, ref in zip(results, valid_texts):
            pred = normalise(result["text"])
            ref  = normalise(ref)
            if len(ref) == 0:
                skipped += 1
                continue
            all_preds.append(pred)
            all_labels.append(ref)

        if (start // batch_size) % 10 == 0:
            done = min(end, total)
            print(f"  Progress: {done}/{total} ({100*done/total:.1f}%)")

    if not all_preds:
        print("  No valid samples found!")
        return None, None

    wer = wer_metric.compute(predictions=all_preds, references=all_labels)
    cer = compute_cer(all_preds, all_labels)

    print(f"\n  ✓ Evaluated {len(all_preds)} samples  (skipped {skipped})")
    print(f"  WER : {wer*100:.2f}%")
    print(f"  CER : {cer*100:.2f}%")

    return wer, cer


def main():
    parser = argparse.ArgumentParser(description="Evaluate ASR models on CoRal test set")
    parser.add_argument("--finetuned_model", default="abargum/whisper-tiny-base",
                        help="Path or HF repo of your fine-tuned Whisper-tiny checkpoint")
    parser.add_argument("--baseline_model",   default="openai/whisper-tiny",
                        help="Baseline Whisper model (default: openai/whisper-tiny)")
    parser.add_argument("--hviske_model",     default="syvai/hviske-v2",
                        help="hviske-v2 model ID (default: syvai/hviske-v2)")
    parser.add_argument("--dataset",          default="CoRal-project/coral-v2",
                        help="HF dataset name")
    parser.add_argument("--dataset_config",   default="read_aloud",
                        help="Dataset config / subset name")
    parser.add_argument("--batch_size",       type=int, default=16)
    parser.add_argument("--output_csv",       default="results.csv",
                        help="Where to write the results table")
    args = parser.parse_args()

    device     = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Device : {device}  |  dtype: {torch_dtype}")

    print(f"\nLoading test split of '{args.dataset}' ({args.dataset_config}) …")
    test_ds = load_dataset(
        args.dataset,
        args.dataset_config,
        split="test",
        trust_remote_code=True,
    )
    test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16_000))
    print(f"Test set size: {len(test_ds)} samples")

    models_to_eval = [
        ("Baseline Whisper-tiny",   args.baseline_model,  "generic"),
        ("Fine-tuned Whisper-tiny", args.finetuned_model, "finetuned"),
        ("hviske-v2",               args.hviske_model,    "generic"),
    ]

    results = []

    for label, model_id, kind in models_to_eval:
        print(f"\nLoading model: {label}  →  {model_id}")
        try:
            if kind == "finetuned":
                pipe = build_finetuned_pipeline(model_id, device, torch_dtype)
            else:
                pipe = build_pipeline(model_id, device, torch_dtype)
        except Exception as e:
            print(f"  [ERROR] Could not load {label}: {e}")
            results.append({"model": label, "wer": "ERROR", "cer": "ERROR"})
            continue

        wer, cer = evaluate_pipeline(pipe, test_ds, args.batch_size, label)

        results.append({
            "model": label,
            "wer":   f"{wer*100:.2f}%" if wer is not None else "N/A",
            "cer":   f"{cer*100:.2f}%" if cer is not None else "N/A",
        })

        del pipe
        torch.cuda.empty_cache()

    print("\n" + "="*50)
    print(f"{'Model':<30} {'WER':>8} {'CER':>8}")
    print("-"*50)
    for r in results:
        print(f"{r['model']:<30} {r['wer']:>8} {r['cer']:>8}")
    print("="*50)

    csv_path = Path(args.output_csv)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "wer", "cer"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()