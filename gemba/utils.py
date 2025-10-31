import ipdb
import pandas as pd
import diskcache as dc
from gemba.gpt_api import GptApi
from gemba.gemba_mqm_utils import TEMPLATE_GEMBA_MQM, apply_template, parse_mqm_answer
from gemba.gemba_esa import TEMPLATE_GEMBA_ESA_ERROR_SPANS, TEMPLATE_GEMBA_ESA_RANKING
from gemba.prompt import prompts, validate_number


def get_gemba_scores(source, hypothesis, source_lang, target_lang, method, model, reference=None):
    # Validate reference usage
    method_uses_ref = method.endswith('_ref')
    if method_uses_ref and reference is None:
        raise ValueError(f"Method '{method}' requires a reference, but none was provided. "
                        f"Please provide a reference file using the --reference flag.")
    if not method_uses_ref and reference is not None:
        print(f"Warning: Reference provided but method '{method}' does not use references. "
              f"Consider using '{method}_ref' to utilize the reference in evaluation.")
    
    # Build DataFrame with source and hypothesis
    df = pd.DataFrame({'source_seg': source, 'target_seg': hypothesis})
    
    # Add reference if provided
    if reference is not None:
        if len(reference) != len(source):
            raise ValueError(f"Reference has {len(reference)} lines but source has {len(source)} lines. "
                           f"All files must have the same number of lines.")
        df['reference_seg'] = reference
    
    df['source_lang'] = source_lang
    df['target_lang'] = target_lang

    cache = dc.Cache(f'cache/{model}_{method}', expire=None, size_limit=int(10e10), cull_limit=0, eviction_policy='none')
    gptapi = GptApi()

    if method == "GEMBA-MQM":
        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_MQM, x), axis=1)
        parse_answer = lambda x: parse_mqm_answer(x, list_mqm_errors=False, full_desc=True)
        answers = gptapi.bulk_request(df, model, parse_answer, cache=cache, max_tokens=500)
    elif method in ["GEMBA-DA", "GEMBA-DA_ref", "GEMBA-SQM", "GEMBA-SQM_ref", "GEMBA-stars", "GEMBA-stars_ref", "GEMBA-classes", "GEMBA-classes_ref"]:
        df["prompt"] = df.apply(lambda x: apply_template(prompts[method]['prompt'], x), axis=1)
        parse_answer = prompts[method]["validate_answer"]
        answers = gptapi.bulk_request(df, model, parse_answer, cache=cache, max_tokens=500)
    elif method == "GEMBA-ESA":
        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_ESA_ERROR_SPANS, x), axis=1)
        parse_answer = lambda x: x
        error_spans = gptapi.bulk_request(df, model, parse_answer, cache=cache)
        df['error_spans'] = pd.DataFrame(error_spans)['answer']

        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_ESA_RANKING, x), axis=1)
        parse_answer = validate_number
        answers = gptapi.bulk_request(df, model, parse_answer, cache=cache)
    else:
        raise Exception(f"Method {method} not supported.")

    return list(pd.DataFrame(answers)['answer'])
