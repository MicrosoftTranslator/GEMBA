import os
import sys
import ipdb
import pandas as pd
import diskcache as dc
from gemba.gpt_api import GptApi
from gemba.CREDENTIALS import credentials
from gemba.gemba_mqm_utils import TEMPLATE_GEMBA_MQM, apply_template, parse_mqm_answer

from absl import app, flags

flags.DEFINE_string('source', None, 'Filepath to the source file.')
flags.DEFINE_string('hypothesis', None, 'Filepath to the translation file.')
flags.DEFINE_string('source_lang', None, 'Source language name.')
flags.DEFINE_string('target_lang', None, 'Target language name.')
flags.DEFINE_bool('verbose', False, 'Verbose mode.')


model = "gpt-4"

def main(argv):
    FLAGS = flags.FLAGS
    assert FLAGS.source is not None, "Source file must be provided."
    assert FLAGS.hypothesis is not None, "Hypothesis file must be provided."

    # check that source and hypothesis files exists
    if not os.path.isfile(FLAGS.source):
        print(f"Source file {FLAGS.source} does not exist.")
        sys.exit(1)
    if not os.path.isfile(FLAGS.hypothesis):
        print(f"Hypothesis file {FLAGS.hypothesis} does not exist.")
        sys.exit(1)

    assert FLAGS.source_lang is not None, "Source language name must be provided."
    assert FLAGS.target_lang is not None, "Target language name must be provided."

    # load both files and strip them
    with open(FLAGS.source, 'r') as f:
        source = f.readlines()
    source = [x.strip() for x in source]
    with open(FLAGS.hypothesis, 'r') as f:
        hypothesis = f.readlines()
    hypothesis = [x.strip() for x in hypothesis]

    assert len(source) == len(hypothesis), "Source and hypothesis files must have the same number of lines."

    df = pd.DataFrame({'source_seg': source, 'target_seg': hypothesis})
    df['source_lang'] = FLAGS.source_lang
    df['target_lang'] = FLAGS.target_lang

    df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_MQM, x), axis=1)

    gptapi = GptApi(credentials, verbose=FLAGS.verbose)
    cache = dc.Cache(f'cache/{model}_GEMBA-MQM', expire=None, size_limit=int(10e10), cull_limit=0, eviction_policy='none')


    answers = gptapi.bulk_request(df, model, lambda x: parse_mqm_answer(x, list_mqm_errors=False, full_desc=True), cache=cache, max_tokens=500)
    
    for answer in answers:
        print(answer['answer'])


if __name__ == "__main__":
    app.run(main)
