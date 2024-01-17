import os
import torch

import copy
from typing import Optional, Sequence, Union, Callable, Dict
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from src.rich_utils import rich_theme
from src.dataset_utils import SupervisedDataset, DataCollatorForSupervisedDataset

import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from accelerate.utils import tqdm
import accelerate


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


class SPIN_Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        datasets,
        batch_size,
        accelerator,
        dl_batch_size: int = 2,
        dl_num_workers: int = 0,
        dl_pin_memory: bool = True,
        weight_decay: float = 1e-5,
        learning_rate: float = 5e-4,
        reg_term: float = 0.5,
    ):
        # for progress print
        self.console = Console(theme=rich_theme)

        # Setting up
        self.model_t = model
        # model to update (t+1)
        self.model = copy.deepcopy(model)
        self.tokenizer = tokenizer
        self.smart_tokenizer_and_embedding_resize()
        self.batch_size = batch_size
        self.accelerator = accelerator
        self.optimizer = torch.optim.AdamW(
            self.get_grouped_params(model, wd=weight_decay),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.reg_term = reg_term

        # dataset preprocessing
        self.console.print("Preprocessing dataset...", style="info")
        self.dataset = SupervisedDataset(datasets, tokenizer)

        # dataloader
        # after synethetic data generation we will update the dataset
        self.train_dl = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=DataCollatorForSupervisedDataset(tokenizer),
            batch_size=dl_batch_size,
            num_workers=dl_num_workers,
            pin_memory=dl_pin_memory,
        )

        # self.console.print("Setting up Accelerator...", style="info")
        # self.model, self.optimizer, self.train_dl = accelerator.prepare(
        #    self.model, self.optimizer, self.train_dl
        # )

    def get_grouped_params(self, model, wd):
        params_with_wd, params_without_wd = [], []

        def apply_decay(x):
            return (
                "gated_cross_attn_layer" in x
                and "ff_gate" not in x
                and "attn_gate" not in x
                and "norm" not in x
                and "bias" not in x
            )

        for n, p in model.named_parameters():
            # if p.requires_grad:
            if apply_decay(n):
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)

        return [
            {"params": params_with_wd, "weight_decay": wd},
            {"params": params_without_wd, "weight_decay": 0.0},
        ]

    def smart_tokenizer_and_embedding_resize(
        self,
    ):
        """Resize tokenizer and embedding.

        Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
        """
        special_tokens_dict = dict()
        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if self.tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if self.tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if self.tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model_t.resize_token_embeddings(len(self.tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.model.get_input_embeddings().weight.data
            output_embeddings = self.model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

    def log_prob(self, output, label):
        logits = output.logits[:, :-1, :].contiguous()
        target = label[:, 1:].long().contiguous().to(logits.device)

        mask = target != -100
        gather_target = target.clone()
        gather_target[~mask] = 0

        logp = F.log_softmax(logits, dim=-1)

        logp_label = torch.gather(logp, 2, gather_target.unsqueeze(-1))
        logp_label = logp_label.squeeze(-1)
        logp_label = logp_label[mask]

        logp_sum = torch.mean(logp_label)
        return logp_sum

    def l(self, t):
        # log(1 + exp(−t))
        return torch.log1p(torch.exp(-t))

    def train_one_iter(self, self_play_iter: int = 0, epochs: int = 4):
        self.model.train()
        self.model_t.eval()

        self.console.print("[Trainer Loop] Start training...", style="info")

        for epoch in range(epochs):
            losses = []
            save_dir = f"./checkpoints/{self_play_iter}/{epoch}"

            # progress bar for logging
            progress_bar = tqdm(
                total=len(self.train_dl),
                desc=f"[Epoch {epoch}]",
                position=0,
                leave=True,
            )

            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=None),
                TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
                TimeRemainingColumn(),
                TextColumn(
                    "Loss: [bold red]{task.fields[loss]}"
                ),  # Custom column for loss
                expand=True,
            ) as progress:
                task = progress.add_task(
                    f"[Epoch {epoch}]", total=len(self.train_dl), loss=0
                )

                for batch_idx, batch in enumerate(self.train_dl):
                    # synthetic data generation
                    x = batch["x"]  # (B x L_x x V)
                    with torch.no_grad():
                        synthetic_y_x = self.model_t.generate(
                            inputs=x.to(self.model_t.device),
                            max_new_tokens=1024,
                            top_p=0.9,
                            do_sample=True,
                            use_cache=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )  # (B x (L_x + L_y') x V)

                    synthetic_label = copy.deepcopy(synthetic_y_x)
                    synthetic_label[:, : x.shape[1]] = IGNORE_INDEX

                    # forward model
                    input_ids = batch["input_ids"].cuda()  # (B x (L_x + L_y) x V)
                    labels = batch["labels"].cuda()  # (B x (L_x + L_y) x V)

                    with torch.no_grad():
                        y_pred_t = self.model_t(input_ids)  # (B x (L_x + L_y) x V)
                        y_prime_t = self.model_t(
                            synthetic_y_x
                        )  # (B x (L_x + L_y') x V)

                    y_pred = self.model(input_ids)
                    y_prime_pred = self.model(synthetic_y_x)

                    # log pθ(y|x)
                    logp = self.log_prob(y_pred, labels)
                    # log pθt(y|x)
                    logp_t = self.log_prob(y_pred_t, labels)
                    # log pθ(y′|x)
                    logp_prime = self.log_prob(y_prime_pred, synthetic_label)
                    # log pθt(y′|x)
                    logp_prime_t = self.log_prob(y_prime_t, synthetic_label)

                    # log pθ(y| x) - log pθt(y|x)
                    L_real = logp - logp_t

                    # log pθ(y′|x) - log pθt(y′|x)
                    L_syn = logp_prime - logp_prime_t

                    L_spin = self.l(self.reg_term * L_real - self.reg_term * L_syn)

                    # update
                    self.optimizer.zero_grad()
                    L_spin.backward()
                    self.optimizer.step()

                    # log loss
                    losses.append(L_spin.item())
                    progress.update(task, advance=1, loss=f"{L_spin.item():.4f}")

            # save model
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.model.save_pretrained(save_dir)
            self.console.print(
                f"[Epoch {epoch}] Save model to {save_dir}", style="info"
            )
