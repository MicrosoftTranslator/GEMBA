import sys
from mt_metrics_eval import data
from gemba.mtme_tools import eval_metrics


dataset = "wmt22"
focus_lps = ['en-de', 'en-ru', 'zh-en']
FINAL_MODELS = []
path = "scores/mt-metrics-eval-v2"

eval_sets = {}
for lp in focus_lps:
    print(lp, file=sys.stderr)
    eval_sets[lp] = data.EvalSet(dataset, lp, True, path=path)

appraise_results = eval_metrics(
    eval_sets, focus_lps, ['sys'], primary_only=False, k=0,
    gold_name="mqm", include_domains=False, seg_level_no_avg=True,
    include_human_with_acc=False)
results = appraise_results[list(appraise_results.keys())[0]]

print(f"Accuracy results")
for key in results.keys():
    print(f"{key}\t{results[key][1]:.3f}")

