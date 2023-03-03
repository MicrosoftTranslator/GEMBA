# GEMBA

## Setup

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

## Running GEMBA

```
python main.py
```

## Evaluate scores

```
export PYTHONPATH=mt-metrics-eval:$PYTHONPATH
python evaluate.py
```

## License
GEMBA code and data are released under the [CC BY-SA 4.0 license](https://github.com/MicrosoftTranslator/GEMBA/blob/main/LICENSE.md).

## Paper
You can read more about GEMBA [in our arXiv paper](https://arxiv.org/pdf/2302.14520.pdf).

## How to Cite

    @misc{https://doi.org/10.48550/arxiv.2302.14520,
      doi = {10.48550/ARXIV.2302.14520},
      url = {https://arxiv.org/abs/2302.14520},
      author = {Kocmi, Tom and Federmann, Christian},
      keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
      title = {Large Language Models Are State-of-the-Art Evaluators of Translation Quality},
      publisher = {arXiv},
      year = {2023},
      copyright = {Creative Commons Attribution 4.0 International}
    }







