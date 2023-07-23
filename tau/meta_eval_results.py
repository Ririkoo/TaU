import os
import json
from mt_metrics_eval import data
import pandas as pd
import numpy as np
import sys


def filtered_sys_score(evs, input_sys, seg_score):
    golden = evs.Scores("seg", "mqm")
    sys_names = set(golden) - {evs.std_ref}
    if input_sys not in sys_names:
        return None
    none_idx = [idx for idx, i in enumerate(golden[input_sys]) if i is not None]
    filter_scores = [seg_score[i] for i in none_idx]
    return [np.mean(filter_scores)]


def meta_eval(
    scores,
    evs,
    test_set,
    language_pair,
    include_human=False,
    use_outliers=False,
    avg_type="none",
    thresh=-1,
):
    results = []
    evs_name = f"{test_set}/{language_pair}"
    for _level in ["sys"]:
        num_scores = len(list(scores[_level].values())[0])
        if num_scores == 1:
            level = "sys"
        elif num_scores == len(evs.docs):
            level = "doc"
        elif num_scores == len(evs.src):
            level = "seg"
        else:
            raise ValueError(
                "Number of scores/system (%d) doesn't match any known granularity in "
                "%s/%s" % (num_scores, test_set, language_pair)
            )

        std_scorer = evs.StdHumanScoreName(level)
        gold_name = std_scorer
        gold_scores = evs.Scores(level, gold_name)
        if gold_scores is None:
            raise ValueError("No scores for %s at %s level." % (gold_name, level))

        close_refs = {"refB"} if evs_name == "wmt21.news/en-de" else set()

        sys_names = set(gold_scores) - {evs.std_ref} - close_refs

        if not include_human:
            sys_names -= evs.human_sys_names

        if not use_outliers:
            sys_names -= evs.outlier_sys_names

        avg = "none" if level == "sys" else avg_type
        corr = evs.Correlation(gold_scores, scores[level], sys_names)
        pearson = corr.Pearson(avg != "none", avg == "sys")
        spearman = corr.Spearman(avg != "none", avg == "sys")
        kendall = corr.Kendall(avg != "none", avg == "sys")
        # Always average KendallLike, otherwise it's very slow.
        if thresh == -1:
            thresh = 25 if gold_name == "wmt-raw" else 0
        kendall_like = corr.KendallLike(averaged=True, thresh=thresh)

        corr_type = "averaging" if avg else "pooling"
        res = {
            "expid": None,
            "level": _level,
            "pearson": pearson[0],
            "pearson_p": pearson[1],
            "kendall": kendall[0],
            "kendall_p": kendall[1],
            "kendall_like": kendall_like[0],
            "meta_info": f"{test_set} {language_pair} HT={include_human},{level},scoring {corr.num_sys}/{len(evs.sys_names)},gold={gold_name},{corr.none_count} None, {corr_type} {corr.num_items}x{corr.num_sys} scores ",
        }
        results.append(res)
    return results


def verify():
    file = "/home/nlp2ct01/runzhe/.mt-metrics-eval/mt-metrics-eval-v2/wmt21.news/metric-scores/zh-en/COMET-DA_2020-refB.sys.score"
    scores = {"sys": {}, "seg": {}}
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            sysn, score = line.split("\t")
            scores["sys"][sysn] = [float(score.strip())]
            scores["seg"][sysn] = [float(score.strip())]  # Not used

    evs = data.EvalSet("wmt21.news", "zh-en")
    results = meta_eval(scores, evs, "wmt21.news", "zh-en", include_human=False)
    ht_results = meta_eval(scores, evs, "wmt21.news", "zh-en", include_human=True)
    assert abs(results[0]["pearson"] - 0.511) < 0.001
    assert abs(ht_results[0]["pearson"] - 0.221) < 0.001

    file = "/home/nlp2ct01/runzhe/.mt-metrics-eval/mt-metrics-eval-v2/wmt21.news/metric-scores/en-de/COMET-DA_2020-refC.sys.score"
    scores = {"sys": {}, "seg": {}}
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            sysn, score = line.split("\t")
            scores["sys"][sysn] = [float(score.strip())]
            scores["seg"][sysn] = [float(score.strip())]  # Not used

    evs = data.EvalSet("wmt21.news", "en-de")
    results = meta_eval(scores, evs, "wmt21.news", "en-de", include_human=False)
    ht_results = meta_eval(scores, evs, "wmt21.news", "en-de", include_human=True)
    assert abs(results[0]["pearson"] - 0.814) < 0.001
    assert abs(ht_results[0]["pearson"] - 0.658) < 0.001
    print("meta-eval algorithm pass verification")


if __name__ == "__main__":
    # verify()
    path = sys.argv[1]

    scores = {"sys": {}, "seg": {}}
    files = os.listdir(path)

    test_set = path.split("/")[-2]
    language_pair = path.split("/")[-1]

    assert test_set in ["wmt21.tedtalks", "wmt21.news"]
    assert language_pair in ["en-de", "en-ru", "zh-en"]

    evs = data.EvalSet(test_set, language_pair)
    files = [i for i in files if "json" in i]

    for fn in files:
        with open(f"{path}/{fn}", "r") as f:
            pre_score = json.load(f)
        seg_scores = []
        for i in pre_score[list(pre_score.keys())[0]]:
            seg_scores.append(i["COMET"])
        scores["seg"][fn.replace(".txt.json", "")] = seg_scores
        scores["sys"][fn.replace(".txt.json", "")] = filtered_sys_score(
            evs, fn.replace(".txt.json", ""), seg_scores
        )

    results = pd.DataFrame(meta_eval(scores, evs, test_set, language_pair))
    results["HT"] = False

    if test_set == "wmt21.news":
        ht_results = pd.DataFrame(
            meta_eval(scores, evs, test_set, language_pair, include_human=True)
        )
        ht_results["HT"] = True
        results = pd.concat([results, ht_results])

    results.to_csv(f"{path}/corr_report.csv")
