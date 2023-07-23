import yaml
import os
import fire
from pathlib import Path
from mt_metrics_eval import data


def get_eval_info(test_set, language_pair):
    evs = data.EvalSet(test_set, language_pair)
    gold_scores = evs.Scores("sys", "mqm")
    sys_names = set(gold_scores)
    return sys_names, evs.std_ref


def gen_tau_cmd(config, test_set, language_pair, systems, std_ref):
    mtme_data = config["mtme_data"]
    save_dir = os.path.join(config["save_dir"], test_set, language_pair)

    model = config["model"]
    lr = config["lr"]
    adam_beta1 = config["adam_beta1"]
    adam_beta2 = config["adam_beta2"]
    adam_weight_decay = config["adam_weight_decay"]
    mcd_runs = config["mcd_runs"]
    optimize_paras = config["optimize_paras"]

    bs = config["data"][test_set][language_pair]["batch"]
    steps = config["data"][test_set][language_pair]["steps"]

    commandlines = f"for i in {systems}; do\
        python tau.py --model {model}\
        -s {mtme_data}/{test_set}/sources/{language_pair}.txt  \
        -r {mtme_data}/{test_set}/references/{language_pair}.{std_ref}.txt \
        -t {mtme_data}/{test_set}/system-outputs/{language_pair}/$i \
        --to_json {save_dir}/$i.json \
        --lr {lr} \
        --adam-beta1 {adam_beta1} --adam-beta2 {adam_beta2}\
        --adam-weight-decay {adam_weight_decay} \
        --mc_dropout {mcd_runs} \
        --component {optimize_paras} \
        --adapt-epoch {steps} \
        --batch_size {bs} \
        --post-infer --quiet; done \n"

    return commandlines, save_dir


def run_exps(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    # print(config)
    commandline_all = ""
    for test_set in config["data"].keys():
        for language_pair in config["data"][test_set].keys():
            sys_names, std_ref = get_eval_info(test_set, language_pair)
            sys_names = " ".join([i + ".txt" for i in sys_names])
            commandlines, save_dir = gen_tau_cmd(
                config, test_set, language_pair, sys_names, std_ref
            )
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            commandline_all += commandlines

    script_file = config_file.split("/")[-1].replace(".yaml", ".sh")
    with open(f"run_{script_file}", "w") as file:
        file.write(commandline_all)
    print(f"Successfully wrote the script to 'run_{script_file}'.")


if __name__ == "__main__":
    fire.Fire(run_exps)
