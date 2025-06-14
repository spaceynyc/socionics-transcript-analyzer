import os
import json
from collections import defaultdict
from typing import Dict, List

try:
    import openai
except ImportError as exc:
    raise SystemExit(
        "The 'openai' package is required to run this script. "
        "Install it with 'pip install openai' and ensure network access."
    ) from exc

SOCIONICS_TYPES = [
    "ILE", "SEI", "ESE", "LII",
    "SLE", "IEI", "EIE", "LSI",
    "SEE", "ILI", "LIE", "ESI",
    "LSE", "EII", "IEE", "SLI",
]

MODELS = [
    "gpt-4o",
    "gpt-3.5-turbo",
    "gpt-4-turbo",
]

def call_openai(model: str, transcript: str) -> Dict[str, float]:
    """Call an OpenAI chat model and return a probability map."""
    prompt = (
        "You are a Socionics analyst. "
        "Given the following transcript, return a JSON object mapping each "
        "Socionics type to the probability (0-1) that the speaker matches "
        "that type. Only output JSON and ensure probabilities sum to 1."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": transcript},
    ]
    response = openai.ChatCompletion.create(model=model, messages=messages)
    content = response.choices[0].message.content.strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {}
    return {t: float(data.get(t, 0)) for t in SOCIONICS_TYPES}


def analyze(transcript: str) -> Dict[str, float]:
    """Analyze transcript with multiple models and average their results."""
    heatmap: Dict[str, float] = defaultdict(float)
    for model in MODELS:
        probs = call_openai(model, transcript)
        for t, p in probs.items():
            heatmap[t] += p
    for t in heatmap:
        heatmap[t] /= len(MODELS)
    return dict(heatmap)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Model Socionics Analyzer")
    parser.add_argument("transcript", help="Path to transcript file")
    args = parser.parse_args()

    with open(args.transcript, "r", encoding="utf-8") as f:
        transcript = f.read()

    results = analyze(transcript)
    print("Probabilities (averaged across models):")
    for t in SOCIONICS_TYPES:
        bar = "#" * int(results.get(t, 0) * 20)
        print(f"{t:3} {results.get(t, 0):.2f} {bar}")


if __name__ == "__main__":
    main()
