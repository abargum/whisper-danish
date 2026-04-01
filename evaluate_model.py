import re
import random
import argparse
import torch
from datasets import load_dataset, Audio
from num2words import num2words
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

BASE_MODEL = "openai/whisper-tiny"
NUM_EXAMPLES = 5
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

def transcribe(model, processor, audio_array, sample_rate, device):
    input_features = processor(
        audio_array, sampling_rate=sample_rate, return_tensors="pt"
    ).input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

def load_whisper(path, device):
    processor = WhisperProcessor.from_pretrained(path, language="danish", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(path).to(device)
    model.generation_config.language = "danish"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.eval()
    return model, processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare fine-tuned vs base Whisper for Danish ASR")
    parser.add_argument("--model_path", default="abargum/whisper-tiny-base", help="Path or HF repo of fine-tuned Whisper model")
    parser.add_argument("--num_examples", type=int, default=NUM_EXAMPLES, help="Number of examples to compare")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print(f"Loading fine-tuned model from: {args.model_path}")
    ft_model, ft_processor = load_whisper(args.model_path, device)

    print(f"Loading base model: {BASE_MODEL}")
    base_model, base_processor = load_whisper(BASE_MODEL, device)

    print("\nLoading dataset...")
    dataset = load_dataset("CoRal-project/coral-v2", "read_aloud", split="test[:100]")
    dataset = dataset.cast_column("audio", Audio(decode=False))
    dataset = dataset.map(clean_danish_text, num_proc=4)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    indices = random.sample(range(len(dataset)), args.num_examples)

    print(f"\n{'='*70}")
    print(f"Comparing {args.num_examples} examples — Base vs Fine-tuned Whisper")
    print(f"{'='*70}\n")

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        audio_array = sample["audio"]["array"]
        sample_rate  = sample["audio"]["sampling_rate"]

        base_pred = transcribe(base_model, base_processor, audio_array, sample_rate, device)
        ft_pred   = transcribe(ft_model,   ft_processor,   audio_array, sample_rate, device)

        print(f"Example {i+1}")
        print(f"  Target:      {sample['text']}")
        print(f"  Base:        {base_pred}")
        print(f"  Fine-tuned:  {ft_pred}")
        print()