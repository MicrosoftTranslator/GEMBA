import os
import sys
import ipdb
import pandas as pd
import diskcache as dc
from absl import app, flags
from gemba.utils import get_gemba_scores


flags.DEFINE_string('method', "GEMBA-MQM", 'Which method to use?')
flags.DEFINE_string('model', "gpt-4", 'OpenAI model')
flags.DEFINE_string('source', None, 'Filepath to the source file.')
flags.DEFINE_string('hypothesis', None, 'Filepath to the translation file.')
flags.DEFINE_string('reference', None, 'Filepath to the reference file (optional, required for *_ref methods).')
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
    if FLAGS.reference is not None and not os.path.isfile(FLAGS.reference):
        print(f"Reference file {FLAGS.reference} does not exist.")
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

    # load reference file if provided
    reference = None
    if FLAGS.reference is not None:
        with open(FLAGS.reference, 'r') as f:
            reference = f.readlines()
        reference = [x.strip() for x in reference]
        
        # validate that reference has the same number of lines
        if len(reference) != len(source):
            print(f"Error: Reference file has {len(reference)} lines but source has {len(source)} lines.")
            print("All files must have the same number of lines.")
            sys.exit(1)

    assert len(source) == len(hypothesis), "Source and hypothesis files must have the same number of lines."

    answers = get_gemba_scores(source, hypothesis, FLAGS.source_lang, FLAGS.target_lang, FLAGS.method, FLAGS.model, reference=reference)

    for answer in answers:
        print(answer)

if __name__ == "__main__":
    app.run(main)
