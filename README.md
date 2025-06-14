# Socionics Transcript Analyzer

This repository contains a simple tool for analyzing text transcripts using multiple language models.

## Multi-Model Socionics Analyzer

`multi_model_socionics_analyzer.py` reads a transcript from a file, sends it to several LLMs, and averages their predicted Socionics type probabilities.

### Usage

```bash
python multi_model_socionics_analyzer.py path/to/transcript.txt
```

The script requires an OpenAI API key available in the environment as `OPENAI_API_KEY`.

The output lists each type with its averaged probability and a small bar visualization.
