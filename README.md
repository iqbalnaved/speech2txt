Here is a clean, production-ready `README.md` for your GitHub repository.

---

# Audio Transcription with Speaker Diarization using WhisperX

This repository provides a complete pipeline to convert audio files into structured, paragraph-separated text transcriptions. By leveraging **WhisperX** and **PyAnnote**, it generates precise word-level timestamps and correctly attributes speech to distinct speaker names (diarization).

## Features

* 🗣️ **Speaker Diarization:** Automatically identifies and separates different speakers.
* ⏱️ **Word-Level Alignment:** Syncs timestamps precisely using phoneme alignment models (`wav2vec2`).
* 📜 **Structured Paragraphs:** Formats the output text logically into speaker blocks with readable time formatting `[HH:MM:SS]`.
* ⚡ **GPU Accelerated:** Optimized to run efficiently on an NVIDIA T4 GPU runtime.

---

## Prerequisites & Setup

### 1. Hugging Face Access (Gated Models)

This pipeline utilizes PyAnnote models, which require you to accept their terms of service.

1. Log in to your [Hugging Face](https://huggingface.co/) account.
2. Accept the user conditions for [PyAnnote Speaker Diarization 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).
3. Accept the user conditions for [PyAnnote Segmentation 3.0](https://huggingface.co/pyannote/segmentation-3.0).

### 2. Generate Authentication Token

* Go to [Hugging Face Token Settings](https://huggingface.co/settings/tokens).
* Create a **Classic Read Token** (or a Fine-Grained token with *Read access to contents of all public gated repos you can access* enabled).
* Save this token securely.

---

## Installation

Install the required `whisperX` library directly from GitHub:

```bash
pip install git+https://github.com/m-bain/whisperX.git

```

---

## Usage

Save your Hugging Face token in your environment variables, and run the transcription script.

```python
import os
from google.colab import userdata  # If running in Google Colab
import whisperx
from whisperx.diarize import DiarizationPipeline

# 1. Configurations
device = "cuda"
audio_file = "path_to_your_audio.mp3"
batch_size = 16 
compute_type = "float16" 

# Fetch token from environment or Colab secrets
hf_token = userdata.get('HF_TOKEN') 
os.environ["HF_TOKEN"] = hf_token

# 2. Transcribe Audio
model = whisperx.load_model("large-v3", device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)

# 3. Align Timestamps
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# 4. Speaker Diarization
diarize_model = DiarizationPipeline(model_name="pyannote/speaker-diarization-3.1", token=hf_token, device=device)
diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments, result)

# 5. Format and Save Output
output_path = "Transcription_Output.txt"
with open(output_path, "w", encoding="utf-8") as f:
    current_speaker = None
    paragraph = []

    for segment in result["segments"]:
        speaker = segment.get("speaker", "UNKNOWN_SPEAKER")
        start_time = segment["start"]
        text = segment["text"].strip()
        
        start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d}"
        
        if speaker != current_speaker:
            if paragraph:
                f.write(" ".join(paragraph) + "\n\n")
                paragraph = []
            f.write(f"[{start_str}] {speaker}:\n")
            current_speaker = speaker
            
        paragraph.append(text)
        
    if paragraph:
        f.write(" ".join(paragraph) + "\n")

print(f"Structured transcription saved to {output_path}")

```

---

## Sample Output Format

```text
[00:00:05] SPEAKER_00:
Welcome to the lecture. Today we will discuss chapter nine of the text.

[00:00:14] SPEAKER_01:
Could you clarify if this section will be included on the upcoming midterm exam?

[00:00:19] SPEAKER_00:
Yes, it covers everything from the initial slide up until our summary next Thursday.

```

---

## License

This project relies on OpenAI's Whisper weights and PyAnnote's open-source components. Please check their respective repository licenses for legal details.
