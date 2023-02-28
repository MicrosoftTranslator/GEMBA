from mt_metrics_eval import data
import scipy

######
# Functions in this script are copied from mt-metrics-eval/wmt22_metrics.ipynb
######


def eval_metrics(eval_sets, langs, levels, primary_only, k, gold_name='std',
                 include_domains=True, seg_level_no_avg=False,
                 include_human_with_acc=False):
    """Evaluate all metrics for eval sets, across multiple task settings.

    Args:
      eval_sets: Map from lang-pair to eval_set objects.
      langs: List of language pairs (eg 'en-de') for which to compute results.
      levels: List of levels for which to compute results, allowed elements are
        'sys' and 'seg'.
      primary_only: Include only primary metrics.
      k: Number of boostrap draws. If 0, no significance tests for metric-score
        differences are run, and execution is much faster.
      gold_name: Name of gold scores to use, standard scores if 'std'.
      include_domains: Generate domain-specific results in addition to global
        results.
      seg_level_no_avg: If True, use only the average_by=None setting for segment-
        level correlations
      include_human_with_acc: If True, include human outputs in accuracy tasks.

    Returns:
      Map from task names to metric -> (rank, corr, sig_string) stats.
    """
    results = {}

    # First task is global accuracy, iff more than one language is given.
    if len(langs) > 0:
        evs_list = [eval_sets[lp] for lp in langs]
        main_refs = [{evs.std_ref} for evs in evs_list]
        close_refs = [set() for evs in evs_list]
        if gold_name == 'std':
            gold = evs_list[0].StdHumanScoreName('sys')
        else:
            gold = gold_name
        humans = [True, False] if include_human_with_acc else [False]
        for human in humans:
            taskname = data.MakeTaskName(
                'wmt22', langs, None, 'sys', human, 'none', 'accuracy', k, gold,
                main_refs, close_refs, False, primary_only)
            print(taskname)
            res = data.CompareMetricsWithGlobalAccuracy(
                evs_list, main_refs, close_refs, include_human=human,
                include_outliers=False, gold_name=gold,
                primary_metrics=primary_only,
                domain=None, k=k, pval=0.05)
            results[taskname] = reformat(res)

    # Remaining tasks are specific to language, domain, etc.
    for lp in langs:
        evs = eval_sets[lp]
        main_refs = {evs.std_ref}
        close_refs = set()
        for domain in [None] + (list(evs.domain_names) if include_domains else []):
            for level in levels:
                gold = evs.StdHumanScoreName(level) if gold_name == 'std' else gold_name
                for avg in 'none', 'sys', 'item':
                    if (level == 'sys' or seg_level_no_avg) and avg != 'none':
                        continue
                    for human in True, False:
                        if human == True and len(evs.ref_names) == 1:
                            continue  # Single ref
                        for corr in 'pearson', 'kendall':
                            corr_fcn = {'pearson': scipy.stats.pearsonr,
                                        'kendall': scipy.stats.kendalltau}[corr]
                            taskname = data.MakeTaskName(
                                'wmt22', lp, domain, level, human, avg, corr, k, gold,
                                main_refs, close_refs, False, primary=primary_only)
                            print(taskname)
                            corrs = data.GetCorrelations(
                                evs=evs, level=level, main_refs={evs.std_ref},
                                close_refs=close_refs, include_human=human,
                                include_outliers=False, gold_name=gold_name,
                                primary_metrics=primary_only, domain=domain)
                            metrics, sig_matrix = data.CompareMetrics(
                                corrs, corr_fcn, average_by=avg, k=k, pval=0.05)
                            # Make compatible with accuracy results.
                            metrics = {evs.DisplayName(m): v for m, v in metrics.items()}
                            results[taskname] = reformat((metrics, sig_matrix))

    return results


def reformat(results):
    """Reformat CompareMetrics() results to match mtme's format."""
    metrics, sig_matrix = results
    res = {}
    for i, (m, (corr, rank)) in enumerate(metrics.items()):
        sigs = ['1' if p < 0.05 else '0' for p in sig_matrix[i]]
        sigs = ['x'] * (i + 1) + sigs[i + 1:]
        res[m] = (rank, corr, ' '.join(sigs))
    return res
