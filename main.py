import diskcache as dc
from gemba.prompt import prompts, language_codes
from gemba.gpt_api import GptApi
from gemba.testset import Testset
from gemba.scores import Scores


def main():
    scenarios = [
        ["text-davinci-003", "GEMBA-DA", [["wmt22", "en-de"], ["wmt22", "zh-en"], ["wmt22", "en-ru"]], ],
        ["text-davinci-003", "GEMBA-DA_ref", [["wmt22", "en-de"], ["wmt22", "zh-en"], ["wmt22", "en-ru"]], ],
    ]

    gptapi = GptApi()
    for scenario in scenarios:
        use_model = scenario[0]
        annotation = scenario[1]
        cache = dc.Cache(f'cache/{use_model}_{annotation}', expire=None, size_limit=int(10e10), cull_limit=0, eviction_policy='none')

        scoring_name = f"{annotation}_{use_model}"

        if use_model not in credentials["deployments"].keys():
            print(f"Model {use_model} not supported by credentials")
            continue

        for dataset, lp in scenario[2]:
            testset = Testset("mt-metrics-eval-v2", dataset, lp)
            if prompts[annotation]["use_ref"]:
                refname = testset.main_ref
            else:
                refname = None

            scores = Scores(scoring_name, testset, refname)

            # starts with -1 as it is incremented before the first request
            hypothesis_index = -1
            total = testset.segments_count()
            for src, hyp, ref, system in testset.iterate_over_all(refname):
                hypothesis_index += 1

                if scores.get_score(system, hypothesis_index) != 'None':
                    continue

                print(f"Processing hypothesis {hypothesis_index}/{total} for {scoring_name} on {dataset}/{lp}")

                data = {
                    "source_seg": src,
                    "target_seg": hyp,
                    "reference_seg": ref,
                    "source_lang": language_codes[lp.split("-")[0]],
                    "target_lang": language_codes[lp.split("-")[1]],
                }
                prompt = prompts[annotation]["prompt"].format(**data)
                parsed_answers = gptapi.request(prompt, use_model, prompts[annotation]["validate_answer"], cache=cache)

                scores.assign_score(system, hypothesis_index, parsed_answers[0]['answer'], parsed_answers[0]['temperature'])

            scores.save()


if __name__ == '__main__':
  main()
