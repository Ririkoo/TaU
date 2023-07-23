#!/usr/bin/env python3
import itertools
import json
import os
from typing import Union

import numpy as np
import torch
from comet.download_utils import download_model
from comet.models import available_metrics, load_from_checkpoint
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr
from pytorch_lightning import seed_everything
from sacrebleu.utils import get_reference_files, get_source_file
from comet.modules import LayerwiseAttention
from typing import Dict, List, Optional, Tuple, Union


def collect_params(model, component):
    params = []
    names = []
    for nm, m in model.named_modules():
        if "ln" in component:
            if isinstance(m, LayerwiseAttention):
                for np_p, p in m.named_parameters():
                    params.append(p)
                    names.append(f"{nm}.{np_p}")
        if "norm" in component:
            if isinstance(m, torch.nn.LayerNorm):
                for np_p, p in m.named_parameters():
                    params.append(p)
                    names.append(f"{nm}.{np_p}")
        if "estimator" in component:
            if "estimator.ff" in nm:
                for np_p, p in m.named_parameters():
                    params.append(p)
                    names.append(f"{nm}.{np_p}")
    return params, names


def configure_model(model, component):
    # enable training mode, disable gradient
    model.train()
    model.requires_grad_(False)
    for nm, m in model.named_modules():
        if "ln" in component:
            if isinstance(m, LayerwiseAttention):
                m.requires_grad_(True)
        if "norm" in component:
            if isinstance(m, torch.nn.LayerNorm):
                m.requires_grad_(True)
        if "estimator" in component:
            if "estimator.ff" in nm:
                m.requires_grad_(True)
    return model

class Tau(torch.nn.Module):
    def __init__(self, model, optimizer, steps=1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0

    @torch.enable_grad()
    @torch.inference_mode(False)
    def forward(
        self,
        samples: List[Dict[str, str]],
        batch_size: int = 16,
        gpus: int = 1,
        mc_dropout: int = 0,
        progress_bar: bool = True,
        accelerator: str = "auto",
        num_workers: int = None,
        length_batching: bool = True,
        save_mcd: bool = False,
        post_infer: bool = False,
    ):

        for _ in range(self.steps):
            outputs = self.model.predict_adapt(
                samples=samples,
                batch_size=batch_size,
                gpus=gpus,
                mc_dropout=mc_dropout,
                progress_bar=progress_bar,
                accelerator=accelerator,
                num_workers=num_workers,
                length_batching=length_batching,
                save_mcd=save_mcd,
                post_infer=post_infer,
                test_optimizer=self.optimizer
            )
        return outputs

def score_command() -> None:
    parser = ArgumentParser(description="Command for scoring MT systems.")
    parser.add_argument("-s", "--sources", type=Path_fr)
    parser.add_argument("-t", "--translations", type=Path_fr, nargs="+")
    parser.add_argument("-r", "--references", type=Path_fr)
    parser.add_argument("-d", "--sacrebleu_dataset", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--quiet", action="store_true", help="Prints only the final system score."
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="ddp",
        choices=["dp", "ddp"],
        help="Pytorch Lightnining accelerator for multi-GPU.",
    )
    parser.add_argument(
        "--to_json",
        type=str,
        default="",
        help="Exports results to a json file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="wmt20-comet-da",
        help="COMET model to be used.",
    )
    parser.add_argument(
        "--model_storage_path",
        help=(
            "Path to the directory where models will be stored. "
            + "By default its saved in ~/.cache/torch/unbabel_comet/"
        ),
        default=None,
    )
    parser.add_argument(
        "--mc_dropout",
        type=Union[bool, int],
        default=False,
        help="Number of inference runs for each sample in MC Dropout.",
    )
    parser.add_argument(
        "--save_mcd_score",
        action="store_true",
        default=False,
        help="Save scores of MC Dropout.",
    )
    parser.add_argument(
        "--seed_everything",
        help="Prediction seed.",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--num_workers",
        help="Number of workers to use when loading data.",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--disable_bar", action="store_true", help="Disables progress bar."
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
        help="Disables sentence embeddings caching. This makes inference slower but saves memory.",
    )
    parser.add_argument(
        "--disable_length_batching",
        action="store_true",
        help="Disables length batching. This makes inference slower.",
    )
    parser.add_argument(
        "--print_cache_info",
        action="store_true",
        help="Print information about COMET cache.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adapt-epoch", type=int, default=1)
    parser.add_argument("--component", default="ln", nargs="+")
    parser.add_argument("--post-infer", default=False, action="store_true")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=0)


    cfg = parser.parse_args()
    seed_everything(cfg.seed_everything)
    if cfg.sources is None and cfg.sacrebleu_dataset is None:
        parser.error(f"You must specify a source (-s) or a sacrebleu dataset (-d)")

    if cfg.sacrebleu_dataset is not None:
        if cfg.references is not None or cfg.sources is not None:
            parser.error(
                f"Cannot use sacrebleu datasets (-d) with manually-specified datasets (-s and -r)"
            )

        try:
            testset, langpair = cfg.sacrebleu_dataset.rsplit(":", maxsplit=1)
            cfg.sources = Path_fr(get_source_file(testset, langpair))
            cfg.references = Path_fr(get_reference_files(testset, langpair)[0])
        except ValueError:
            parser.error(
                "SacreBLEU testset format must be TESTSET:LANGPAIR, e.g., wmt20:de-en"
            )
        except Exception as e:
            import sys

            print("SacreBLEU error:", e, file=sys.stderr)
            sys.exit(1)

    if cfg.model.endswith(".ckpt") and os.path.exists(cfg.model):
        model_path = cfg.model
    elif cfg.model in available_metrics:
        model_path = download_model(cfg.model, saving_directory=cfg.model_storage_path)
    else:
        parser.error(
            "{} is not a valid checkpoint path or model choice. Choose from {}".format(
                cfg.model, available_metrics.keys()
            )
        )
    model = load_from_checkpoint(model_path)
    # model.eval()

    if model.requires_references() and (cfg.references is None):
        parser.error(
            "{} requires -r/--references or -d/--sacrebleu_dataset.".format(cfg.model)
        )

    if not cfg.disable_cache:
        model.set_embedding_cache()

    with open(cfg.sources(), encoding="utf-8") as fp:
        sources = [line.strip() for line in fp.readlines()]

    translations = []
    for path_fr in cfg.translations:
        with open(path_fr(), encoding="utf-8") as fp:
            translations.append([line.strip() for line in fp.readlines()])

    if cfg.references is not None:
        with open(cfg.references(), encoding="utf-8") as fp:
            references = [line.strip() for line in fp.readlines()]
        data = {
            "src": [sources for _ in translations],
            "mt": translations,
            "ref": [references for _ in translations],
        }
    else:
        data = {"src": [sources for _ in translations], "mt": translations}

    # configure model for test-time behaviors
    model = configure_model(model, component=cfg.component)  # set grad
    params, param_names = collect_params(model, component=cfg.component)
    print(f"optimized params: {param_names}")
    optimizer = torch.optim.Adam(params, lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.adam_weight_decay)
    print(f"optmizer: {optimizer}")
    model = Tau(model, optimizer, steps=cfg.adapt_epoch)

    if cfg.gpus > 1 and cfg.accelerator == "ddp":
        raise NotImplementedError
    else:
        # If not using Multiple GPUs we will score each system independently
        # to maximize cache hits!
        seg_scores, std_scores, sys_scores, mcd_scores = [], [], [], []
        new_data = []
        for i in range(len(cfg.translations)):
            sys_data = {k: v[i] for k, v in data.items()}
            sys_data = [dict(zip(sys_data, t)) for t in zip(*sys_data.values())]
            new_data.append(np.array(sys_data))
            # wrap predict forward
            outputs = model(
                samples=sys_data,
                batch_size=cfg.batch_size,
                gpus=cfg.gpus,
                mc_dropout=cfg.mc_dropout,
                progress_bar=(not cfg.disable_bar),
                accelerator=cfg.accelerator,
                num_workers=cfg.num_workers,
                length_batching=(not cfg.disable_length_batching),
                save_mcd=cfg.save_mcd_score,
                post_infer=cfg.post_infer,
            )
            if len(outputs) == 3:
                seg_scores.append(outputs["scores"])
                std_scores.append(outputs["metadata"]["mcd_std"])
                if "mcd_outputs" in outputs["metadata"].keys():
                    mcd_scores.append(outputs["metadata"]["mcd_outputs"])
            else:
                seg_scores.append(outputs[0])
                std_scores.append(None)

            sys_scores.append(sum(outputs[0]) / len(outputs[0]))
        data = new_data

    files = [path_fr.rel_path for path_fr in cfg.translations]
    data = {file: system_data.tolist() for file, system_data in zip(files, data)}

    for i in range(len(data[files[0]])):  # loop over (src, ref)
        for j in range(len(files)):  # loop of system
            data[files[j]][i]["COMET"] = seg_scores[j][i]
            if cfg.mc_dropout:
                data[files[j]][i]["variance"] = std_scores[j][i]

                data[files[j]][i]["mcd_scores"] = mcd_scores[j][i] if len(mcd_scores[j][i]) > 1 else None
                if not cfg.quiet:
                    print(
                        "{}\tSegment {}\tscore: {:.4f}\tvariance: {:.4f}".format(
                            files[j], i, seg_scores[j][i], std_scores[j][i]
                        )
                    )
            else:
                if not cfg.quiet:
                    print(
                        "{}\tSegment {}\tscore: {:.4f}".format(
                            files[j], i, seg_scores[j][i]
                        )
                    )

    for j in range(len(files)):
        print("{}\tscore: {:.4f}".format(files[j], sys_scores[j]))

    if cfg.to_json != "":
        with open(cfg.to_json, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        print("Predictions saved in: {}.".format(cfg.to_json))

    if cfg.print_cache_info:
        print(model.retrieve_sentence_embedding.cache_info())


if __name__ == "__main__":
    score_command()
