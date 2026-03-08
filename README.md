# 🌊✨ AuraFlow

> **Multimodal Video Intelligence Platform** — Transcribe, Analyze, Compare, and Understand any video using AI.

AuraFlow is an end-to-end video analysis pipeline built on Google Colab that combines **speech recognition**, **speaker diarization**, **visual understanding**, **fine-tuned multimodal sentiment analysis**, **semantic search**, and **similarity evaluation** into a single Gradio web interface.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🎙️ **Transcription** | Whisper-powered speech-to-text with speaker diarization (pyannote) |
| 👥 **Speaker Detection** | Identifies and labels individual speakers with timestamps |
| 🎨 **Visual Analysis** | BLIP keyframe extraction and captioning synced to transcript |
| 📋 **Combined Summary** | LLaMA merges audio + visual into one cohesive overview |
| 📑 **Chapters & Key Moments** | Auto-generated chapter breakdown and highlights |
| 🎭 **Multimodal Sentiment Timeline** | Fine-tuned RoBERTa (85%+ accuracy) fused with BLIP visual emotion scoring |
| 🎯 **Similarity Evaluation** | Auto mode (LLaMA generates reference) + Manual mode with semantic match score |
| 🌍 **Translation** | Translate transcript and summaries into 14 languages |
| 💬 **Q&A Chat** | RAG + LLaMA — ask anything about the video |

---

## 🧠 How It Works

```
Video Input
    │
    ├──► FFmpeg ──► Whisper ──► Transcript + Speaker Labels
    │                               │
    │                               ├──► FAISS RAG Index (Q&A)
    │                               ├──► LLaMA → Summary, Chapters, Key Moments
    │                               ├──► Fine-tuned RoBERTa → Text Sentiment
    │                               └──► Google Translate → Multilingual Output
    │
    └──► OpenCV Keyframes ──► BLIP Captions ──► Visual Emotion Scoring
                                                        │
                                            Text + Visual Fusion
                                                        │
                                            Multimodal Sentiment Timeline
                                                        │
                                            LLaMA → Combined Summary
                                                        │
                                            Embedder + LLaMA → Similarity Score
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Speech Recognition | [OpenAI Whisper](https://github.com/openai/whisper) |
| Speaker Diarization | [pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) |
| Visual Captioning | [Salesforce BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) |
| Language Model | [LLaMA 3.3 70B via Groq](https://console.groq.com) |
| Sentiment Model | Fine-tuned RoBERTa on Kaggle Twitter dataset (85%+ accuracy) |
| Visual Emotion Fusion | BLIP frame descriptions scored and fused with RoBERTa output |
| Semantic Embeddings | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Sentiment Visualization | [Plotly](https://plotly.com/python/) interactive charts + confusion matrix |
| Vector Search | [FAISS](https://github.com/facebookresearch/faiss) |
| Translation | [deep-translator (Google)](https://github.com/nidhaloff/deep-translator) |
| UI | [Gradio](https://gradio.app) |
| Runtime | Google Colab (GPU) |

---

## ⚙️ Setup & Usage

### Prerequisites
- Google Colab account (free tier works, T4 GPU recommended)
- [Groq API Key](https://console.groq.com) — free tier
- [HuggingFace Token](https://huggingface.co/settings/tokens) — free account
- Kaggle Twitter Sentiment dataset (`twitter_training.csv`) — uploaded once, saved to Google Drive automatically
- Accept pyannote model terms at:
  - https://hf.co/pyannote/speaker-diarization-3.1
  - https://hf.co/pyannote/segmentation-3.0

### Run Order

```
Cell 1  →  Bootstrap (pins numpy+scipy, auto-restarts kernel)
Cell 2  →  Install all dependencies → manually restart runtime
Cell 3  →  Upgrade torchvision
Cell 4  →  Mount Google Drive + load/upload Twitter dataset
Cell 5  →  Pin huggingface_hub + transformers versions
Cell 6  →  Numpy fix (run only if AttributeError appears)
Cell 7  →  Verify all packages installed
Cell 8  →  Set GROQ_API_KEY and HF_TOKEN
Cell 9  →  Load all models (Whisper, BLIP, pyannote, Groq, Embedder)
Cell 10 →  Audio + transcription functions
Cell 11 →  Visual pipeline functions
Cell 12 →  RAG, translation, intelligence functions
Cell 13 →  Similarity evaluation (Auto + Manual modes)
Cell 14 →  Fine-tune RoBERTa + BLIP visual emotion fusion
Cell 15 →  Launch Gradio UI
```

> ⚠️ **Important:** After Cell 1 auto-restarts, skip it and start from Cell 2. After Cell 2 finishes, manually restart the runtime once more before continuing.

---

## 🎭 Multimodal Sentiment Analysis

The most technically advanced feature — combines **text** and **visual** signals for richer sentiment understanding:

**Text Sentiment — Fine-tuned RoBERTa:**
- Fine-tuned on the Kaggle Twitter Sentiment dataset
- Achieves **85%+ accuracy** on 3-class classification (Positive / Neutral / Negative)
- Trained with fixed seed (42) for reproducibility
- Outputs calibrated class probabilities per transcript segment

**Visual Emotion Fusion — BLIP:**
- BLIP keyframe descriptions are independently scored for emotional tone
- Visual scores are fused with RoBERTa text scores per segment
- Final sentiment reflects both **what was said** and **what was seen**

**Visualization:**
- Interactive Plotly area chart — score per segment over time
- Color coded: 🟢 Positive · 🔴 Negative · ⚫ Neutral
- Confusion matrix and full classification report included

---

## 🎯 Similarity Evaluation

Two modes for flexible video content assessment:

**Auto Mode:**
- LLaMA 3.3 automatically generates a reference text directly from the transcript
- Compares it against the combined summary — fully automated, no manual input needed

**Manual Mode:**
- Paste your own reference text (expected answer / ideal description)
- Compares against the video's combined summary (audio + visual)
- Returns a semantic match score (%) using cosine similarity on sentence embeddings
- Shows top 3 most relevant transcript segments
- LLaMA explains what matched and what was missing

---

## 💰 Cost

| Service | Cost |
|---|---|
| Groq API | Free (14,400 req/day) |
| HuggingFace | Free |
| Google Colab | Free (T4 GPU) |
| Kaggle Dataset | Free |
| **Total** | **$0** |

---

## 📁 Project Structure

```
AuraFlow_Integrated.ipynb    # Main notebook — all cells
twitter_training.csv         # Kaggle Twitter dataset (upload once to Google Drive)
README.md                    # This file
```

---

## 🏆 Built For

This project was built for a hackathon targeting automated video content evaluation — comparing spoken and visual content against reference answers using semantic AI, with multimodal sentiment analysis powered by a fine-tuned RoBERTa transformer model fused with visual signals from BLIP.

---

## 📌 Notes

- The Gradio share link is active only while the Colab session is running
- For permanent deployment, use [Hugging Face Spaces](https://huggingface.co/spaces) via `gradio deploy`
- The Twitter dataset is saved to Google Drive after the first upload — subsequent runs load it automatically without re-uploading
- All dependency conflict warnings during install are from Colab's pre-installed packages and can be safely ignored
- **Never push your API keys to GitHub** — replace them in Cell 8 before committing
