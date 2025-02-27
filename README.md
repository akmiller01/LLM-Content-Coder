# LLM-Content-Coder
A General Purpose LLM Content Coder, utilizing structured outputs

## Installation (windows)

```bash
python -m virtualenv venv
cd venv/Scripts
activate
cd ..
pip install -r requirements.txt
```

## Example usage
```bash
python classify.py -f example_data/fintech_short.csv -i 7 -c "Alternative lending" "Capital markets" "CFO stack" "Commercial finance" "Financial services infrastructure" "Payments" "Regtech" "Wealthtech" -o example_data/fintech_short_gemini_single.csv

python classify.py -f example_data/fintech_short.csv -i 7 -m -c "Alternative lending" "Capital markets" "CFO stack" "Commercial finance" "Financial services infrastructure" "Payments" "Regtech" "Wealthtech" -o example_data/fintech_short_gemini_multi.csv
```

## Models available
- `-d gemini`: Default model. Gemini 2.0 Flash. Capable but API is a little unstable.
- `-d gpt`: OpenAI GPT-4o-mini. A small MoE model that is reasonably capable with more robust API infrastructure.
- `-d {any_ollama_model}`: E.g. `phi4` (14B), `llama3.1:8b` (8B), or `mistral` (7B). Will require high RAM (4-11 GB free) and be slower to run on CPU.