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

### Evaluating script with GEMBA-MQM

It assume two files with the same number of lines. It prints the score for each line pair:

```
python gemba_mqm.py --source=source.txt --hypothesis=hypothesis.txt --source_lang=English --target_lang=Czech
```


### Collecting experiments for GEMBA-DA

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
You can read more about GEMBA-DA [in our arXiv paper](https://arxiv.org/pdf/2302.14520.pdf) 
or GEMBA-MQM [in our arXiv paper](https://arxiv.org/pdf/2310.13988.pdf).

## How to Cite


### GEMBA-MQM 

    @inproceedings{kocmi-federmann-2023-gemba-mqm,
        title = {GEMBA-MQM: Detecting Translation Quality Error Spans with GPT-4},
        author = {Kocmi, Tom  and Federmann, Christian},
        booktitle = "Proceedings of the Eighth Conference on Machine Translation",
        month = dec,
        year = "2023",
        address = "Singapore",
        publisher = "Association for Computational Linguistics",
    }

### GEMBA-DA

    @inproceedings{kocmi-federmann-2023-large,
        title = "Large Language Models Are State-of-the-Art Evaluators of Translation Quality",
        author = "Kocmi, Tom and Federmann, Christian",
        booktitle = "Proceedings of the 24th Annual Conference of the European Association for Machine Translation",
        month = jun,
        year = "2023",
        address = "Tampere, Finland",
        publisher = "European Association for Machine Translation",
        url = "https://aclanthology.org/2023.eamt-1.19",
        pages = "193--203",
    }







