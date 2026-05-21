# AuraFlow 

> **Multimodal Video Intelligence** — Transcription · Speaker Diarization · Visual Analysis · Sentiment · RAG Q&A · Translation · Summary Video

AuraFlow is an end-to-end video understanding pipeline that processes any uploaded video and extracts rich multimodal intelligence from it — combining speech, vision, and language models into a single Gradio interface.

---

## What It Does

Upload a video. AuraFlow does the rest:

- **Transcribes** speech using OpenAI Whisper with word-level timestamps
- **Identifies speakers** using pyannote speaker diarization
- **Describes frames** using LLaMA 3.2 Vision (via Groq) for accurate scene descriptions
- **Detects emotions** in video frames using CLIP (ViT-L-14) zero-shot classification
- **Analyzes sentiment** across the video timeline by fusing fine-tuned RoBERTa (audio) + CLIP (visual)
- **Summarizes** the video combining both audio transcript and visual narrative
- **Generates chapters** and key moment markers
- **Answers questions** about the video using FAISS-backed RAG + LLaMA 3.3
- **Translates** transcript and summaries into 14 languages
- **Creates a summary video** from motion-detected keyframes
- **Evaluates similarity** between the video summary and a reference (auto or manual)
- **Maps places** mentioned in the transcript via Google Maps embed
- **Builds a semantic network** of transcript segments showing content relationships

---

## Models Used

| Task | Model |
|---|---|
| Speech-to-text | OpenAI Whisper (`base`) |
| Speaker diarization | pyannote `speaker-diarization-3.1` |
| Frame description | LLaMA 3.2 Vision (`llama-4-scout-17b-16e-instruct`) via Groq |
| Visual emotion detection | CLIP `ViT-L-14` (OpenCLIP, zero-shot) |
| Text sentiment | Fine-tuned RoBERTa (`twitter-roberta-base-sentiment-latest`) |
| Summarization / Q&A / Chapters | LLaMA 3.3 70B (`llama-3.3-70b-versatile`) via Groq |
| Sentence embeddings (RAG) | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Translation | Google Translate (deep-translator) |
| Vector search | FAISS (flat inner product index) |

---

## Architecture

```
Video Upload
    │
    ├── [Audio track] ──────────────────────────────────────────────────┐
    │       │                                                            │
    │   Whisper ASR                                                      │
    │       │                                                            │
    │   pyannote Diarization                                             │
    │       │                                                            │
    │   Transcript (segments + speaker labels)                           │
    │       │                                                            │
    │       ├── LLaMA 3.3 → Summary / Chapters / Key Moments            │
    │       ├── FAISS + MiniLM → RAG Index (Q&A)                        │
    │       ├── Fine-tuned RoBERTa → Text Sentiment (per segment)        │
    │       └── deep-translator → 14-language Translation               │
    │                                                                    │
    ├── [Video track] ───────────────────────────────────────────────────┤
    │       │                                                            │
    │   Keyframe Extraction (motion-diff or uniform sampling)            │
    │       │                                                            │
    │       ├── LLaMA 3.2 Vision → Scene Descriptions                   │
    │       ├── CLIP ViT-L-14 → Emotion Labels (27 fine-grained)        │
    │       └── FFmpeg → Summary Video (keyframe clips stitched)        │
    │                                                                    │
    └── [Fusion] ────────────────────────────────────────────────────────┘
            │
            ├── LLaMA 3.3 → Combined Audio + Visual Summary
            ├── RoBERTa (audio) + CLIP (visual) → Multimodal Sentiment Timeline
            ├── MiniLM cosine similarity → Similarity Evaluation
            └── Gradio UI → All outputs rendered in tabbed interface
```

---

## Sentiment Pipeline

The multimodal sentiment system runs at **2-second resolution** across the full video:

**Audio channel** — Each transcript segment is scored by the fine-tuned RoBERTa model (Negative / Neutral / Positive, with a continuous 0–1 score). The model was fine-tuned on the Kaggle Twitter Entity Sentiment dataset (~24,000 balanced samples, 5 epochs) and achieves **85%+ accuracy** per class.

**Visual channel** — CLIP zero-shot classifies each keyframe against 27 fine-grained emotion labels (e.g. "genuine smile", "frustration", "anxiety", "pride"). Scores are weighted into a Positive / Negative / Neutral bucket with a 0–1 sentiment score.

**Fusion** — Both channels contribute at full weight (100% each). The fused score is their average. The timeline chart shows all three lines:
- 🟣 Dashed — Audio (RoBERTa)
- 🟡 Dotted — Visual (CLIP)
- 🔵 Solid — Combined (fused)

---

## Similarity Evaluation

Two modes for evaluating how well the generated summary represents the video:

**Auto mode** — LLaMA 3.3 generates a clean reference paragraph from the raw transcript (audio) and a separate one from the visual narrative. Cosine similarity (via MiniLM embeddings) is computed between:
- Audio summary vs audio reference
- Visual narrative vs visual reference
- Combined summary vs averaged reference

All scores are capped at 95% to stay realistic. LLaMA 3.3 then writes a qualitative explanation of what the summary covers and what it misses.

**Manual mode** — Paste any reference text. MiniLM computes cosine similarity against the combined summary. Top-3 most relevant transcript segments are surfaced alongside LLaMA's analysis.

---

## Summary Video Accuracy

After generating a summary video, ROUGE scoring (ROUGE-1, ROUGE-2, ROUGE-L) measures how well the selected clips represent the full transcript:

- Identifies which transcript segments fall inside the selected clip windows
- Computes ROUGE F1 between the full transcript text and the covered text
- Reports coverage %, per-metric scores, and a grade (Excellent / Good / Fair / Poor)

---

## Semantic Network

Builds a graph of transcript segments where edges connect semantically similar segments (cosine similarity above a configurable threshold):

- **Purple nodes** — Major nodes (high degree, core topics)
- **Amber nodes** — Useful leaf nodes (low degree but informative)
- **Red X nodes** — Useless filler (identified by LLaMA)
- Leaf nodes connect to their nearest major node via dotted lines

The "Summarize from Semantic Net" button sends major + useful leaf texts to LLaMA for a graph-informed summary focused only on the most semantically central content.

---

## Setup

### Requirements

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install openai-whisper
pip install "transformers==4.47.0" "tokenizers==0.21.0" "accelerate>=0.26" "huggingface_hub==0.33.5"
pip install "sentence-transformers>=3.0"
pip install groq faiss-cpu gradio "pandas>=2.0,<3.0"
pip install opencv-python-headless Pillow deep-translator
pip install pyannote.audio
pip install "scikit-learn>=1.6.1" "hdbscan>=0.8.40" "umap-learn>=0.5.6"
pip install open-clip-torch rouge-score
apt-get install ffmpeg
```

**Pinned versions (critical):**
```bash
pip install "numpy==2.2.2" "scipy==1.15.2" --force-reinstall --no-deps
```

### API Keys

Two keys are required before running:

```python
GROQ_API_KEY = ""   # https://console.groq.com
HF_TOKEN     = ""   # https://huggingface.co/settings/tokens
```

The HuggingFace token must have access to [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) — accept the model terms on the model page before running.

### Run Order

The notebook must be executed in this exact order due to kernel restart requirements:

```
Cell 1  → Bootstrap (pins numpy/scipy, restarts kernel — expected)
Cell 2  → Install all packages (restart runtime after)
Cell 3  → Upgrade torchvision
Cell 4  → Pin transformers/huggingface_hub
Cell 5  → NumPy fix (only if AttributeError on _blas_supports_fpe)
Cell 6  → Verify all packages
Cell 7  → Set API keys
Cell 8  → Load all models
Cell 9  → Audio pipeline functions
Cell 10 → Visual pipeline functions
Cell 11 → RAG / translation / summary functions
Cell 12 → Similarity evaluation functions
Cell 13 → Fine-tune RoBERTa sentiment model (~8 min, saved to Drive)
Cell 14 → Launch Gradio UI
```

> **Note:** Cell 13 saves the fine-tuned model checkpoint to Google Drive. On subsequent runs it loads from the checkpoint and skips retraining.

---

## Gradio Interface Tabs

| Tab | What it shows |
|---|---|
| 📝 Transcript | Interactive segment viewer with ✓ ✗ ? accuracy marking |
| 📊 Analysis | Summary, chapters, key moments |
| 🎨 Visual | Visual narrative, audio-visual sync timeline, keyframe gallery |
| 📋 Combined Summary | Integrated audio + visual overview |
| 🎭 Multimodal Sentiment | Dual-channel sentiment timeline chart + emotion stats |
| 🌍 Translation | Full transcript + summary in 14 languages |
| 💬 Q&A Chat | RAG-powered chatbot grounded in the transcript |
| 🎯 Similarity Evaluation | Auto (LLaMA reference) and Manual similarity scoring |
| 🎬 Summary Video | Motion-keyframe video with ROUGE accuracy report |
| 🕸️ Semantic Network | Transcript segment similarity graph + graph-based summary |

---

## Supported Languages (Translation)

Hindi, Spanish, French, German, Portuguese, Japanese, Korean, Chinese (Simplified), Arabic, Italian, Tamil, Telugu, Malayalam, Kannada

---

## Supported Media Types

| Input | Behavior |
|---|---|
| Audio + Video | Full pipeline — all features active |
| Audio only | Transcription + NLP features; visual tabs show "no video" |
| Video only | Visual features + keyframes; transcript tabs show "no audio" |

---

## Limitations

- Runs on Google Colab (GPU recommended; CPU will be significantly slower)
- Whisper `base` model is fast but less accurate than `large` — swap the model name for higher accuracy at the cost of speed
- CLIP emotion classification is zero-shot and works best with clear facial expressions in frame
- Speaker diarization accuracy degrades with overlapping speech or more than ~5 simultaneous speakers
- Translation via Google Translate (free tier) may hit rate limits on very long transcripts
- Summary video generation requires FFmpeg and sufficient `/content/` disk space

---

## Project Structure

```
auraflow_git.ipynb          ← main notebook (all code, run in Colab)
checkpoints/                ← Gradio model checkpoints (auto-created)
results/                    ← Sample output images per epoch (auto-created)
/content/drive/MyDrive/
  ├── auraflow_sentiment_ckpt/   ← fine-tuned RoBERTa weights (saved to Drive)
  └── summary_video.mp4          ← generated summary video
```

---

## Citation / Credits

- [OpenAI Whisper](https://github.com/openai/whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [CLIP / OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Cardiff NLP RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [Groq API](https://console.groq.com) — LLaMA 3.2 Vision + LLaMA 3.3 70B
- [FAISS](https://github.com/facebookresearch/faiss)
- [Gradio](https://gradio.app)
