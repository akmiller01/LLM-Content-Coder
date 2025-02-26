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
python gemini_classify.py -f example_data/fintech_short.csv -i 7 -c "Alternative lending" "Capital markets" "CFO stack" "Commercial finance" "Financial services infrastructure" "Payments" "Regtech" "Wealthtech" -o example_data/fintech_short_gemini_single.csv

python gemini_classify.py -f example_data/fintech_short.csv -i 7 -m -c "Alternative lending" "Capital markets" "CFO stack" "Commercial finance" "Financial services infrastructure" "Payments" "Regtech" "Wealthtech" -o example_data/fintech_short_gemini_multi.csv
```