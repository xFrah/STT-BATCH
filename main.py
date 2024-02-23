import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os

# ensure folders audio and output exist
os.makedirs("audio", exist_ok=True)
os.makedirs("output", exist_ok=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

file_lines = []

audio_files = os.listdir("audio")
for i, audio_file in enumerate(audio_files):
    result = pipe(f"audio/{audio_file}")
    file_lines.append(audio_file.encode("utf-8") + b":")
    for l in result["text"].split("."):
        if len(l) > 0:
            file_lines.append(l.encode("utf-8").strip() + b".")
    file_lines.append(b"")
    print(f"Transcribed {audio_file}, ({i + 1}/{len(audio_files)})")

with open(f"transcript.txt", "wb") as f:
    f.write(b"\n".join(file_lines))
