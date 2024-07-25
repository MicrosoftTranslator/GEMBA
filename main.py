import os
import sys
import ipdb
import pandas as pd
import diskcache as dc
from absl import app, flags
from gemba.gpt_api import GptApi
from gemba.gemba_mqm_utils import TEMPLATE_GEMBA_MQM, apply_template, parse_mqm_answer
from gemba.gemba_esa import TEMPLATE_GEMBA_ESA_ERROR_SPANS, TEMPLATE_GEMBA_ESA_RANKING
from gemba.prompt import prompts, validate_number


flags.DEFINE_string('method', "GEMBA-MQM", 'Which method to use?')
flags.DEFINE_string('model', "gpt-4", 'OpenAI model')
flags.DEFINE_string('source', None, 'Filepath to the source file.')
flags.DEFINE_string('hypothesis', None, 'Filepath to the translation file.')
flags.DEFINE_string('source_lang', None, 'Source language name.')
flags.DEFINE_string('target_lang', None, 'Target language name.')


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

    cache = dc.Cache(f'cache/{FLAGS.model}_{FLAGS.method}', expire=None, size_limit=int(10e10), cull_limit=0, eviction_policy='none')
    gptapi = GptApi()

    if FLAGS.method == "GEMBA-MQM":
        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_MQM, x), axis=1)
        parse_answer = lambda x: parse_mqm_answer(x, list_mqm_errors=False, full_desc=True)
        answers = gptapi.bulk_request(df, FLAGS.model, parse_answer, cache=cache, max_tokens=500)
    elif FLAGS.method in ["GEMBA-DA", "GEMBA-DA_ref", "GEMBA-SQM", "GEMBA-SQM_ref", "GEMBA-stars", "GEMBA-stars_ref", "GEMBA-classes", "GEMBA-classes_ref"]:
        df["prompt"] = df.apply(lambda x: apply_template(prompts[FLAGS.method]['prompt'], x), axis=1)
        parse_answer = prompts[FLAGS.method]["validate_answer"]
        answers = gptapi.bulk_request(df, FLAGS.model, parse_answer, cache=cache, max_tokens=500)
    elif FLAGS.method == "GEMBA-ESA":
        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_ESA_ERROR_SPANS, x), axis=1)
        parse_answer = lambda x: x
        error_spans = gptapi.bulk_request(df, FLAGS.model, parse_answer, cache=cache)
        df['error_spans'] = pd.DataFrame(error_spans)['answer']

        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_ESA_RANKING, x), axis=1)
        parse_answer = validate_number
        answers = gptapi.bulk_request(df, FLAGS.model, parse_answer, cache=cache)
    else:
        raise Exception(f"Method {FLAGS.method} not supported.")

    for answer in answers:
        print(answer['answer'])

if __name__ == "__main__":
    app.run(main)
