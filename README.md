# GEMBA

# Setup

Install required packages with python >= 3.8 

```
pip install -r requirements.txt
```

Get mt-metric-eval and download resources:

```
git clone https://github.com/google-research/mt-metrics-eval.git
cd mt-metrics-eval
pip install .
alias mtme='python3 -m mt_metrics_eval.mtme'
mtme --download
cd ..
mv ~/.mt-metrics-eval/mt-metrics-eval-v2 mt-metrics-eval-v2
```


Update credentials in `CREDENTIALS.py` with your own.

# Running GEMBA

```
python main.py
```

# Evaluate scores

```
export PYTHONPATH=mt-metrics-eval:$PYTHONPATH
python evaluate.py
```
