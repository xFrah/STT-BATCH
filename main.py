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

audio_files = os.listdir("audio")
for i, audio_file in enumerate(audio_files):
    result = pipe(f"audio/{audio_file}")
    os.makedirs("output/" + audio_file, exist_ok=True)
    os.rename(f"audio/{audio_file}", f"output/{audio_file}/{audio_file}")
    with open(f"output/{audio_file}/{audio_file}_transcript.txt", "wb") as f:
        f.write(result["text"].encode("utf-8"))
    print(f"Transcript for {audio_file} saved to output/{audio_file}/{audio_file}_transcript.txt ({i + 1}/{len(audio_files)})")
