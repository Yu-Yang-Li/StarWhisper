---
name: starwhisper-lowsnr-spectra
description: Inspect the low-SNR stellar-spectra-as-language line and list present files. Use when the user asks about LAMOST/PHOENIX tokenized spectra, SNR-stage fine-tuning, Jaredxjc Hugging Face weights, or Low-SNR-Stellar-Spectra-as-Language.
---

# Low-SNR stellar spectra

Folder: [`Low-SNR-Stellar-Spectra-as-Language/`](../../Low-SNR-Stellar-Spectra-as-Language/README.md)  
Standalone repo: https://github.com/Jared-web03/Low-SNR-Stellar-Spectra-as-Language  
Weights: https://huggingface.co/Jaredxjc/Low-SNR-Stellar-Spectra-as-Language

```powershell
python skills/starwhisper-lowsnr-spectra/scripts/inspect_pipeline.py
```

Training code and the `code/` pipeline are public; the curated full tokenized dataset may still be "coming soon". Do not invent a dataset dump path.

Do not treat a generated or denoised spectrum as a new observation. Keep SNR stage labels as given.
