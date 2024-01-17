import os
import copy
import transformers
import torch
import time
from dataclasses import dataclass, field

import joblib
import _pickle as cPickle
import gc
import pickle

from tqdm import tqdm
from typing import Dict, Sequence, List
from torch.utils.data import Dataset


DEFAULT_SYSTEM_PROMPT = "You are helpful language model"
IGNORE_INDEX = -100


class ProgressBarFile(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def __enter__(self):
        self.file = open(self.file_path, "rb")
        self.length = os.fstat(self.file.fileno()).st_size
        self.progress_bar = tqdm(total=self.length, unit="iB", unit_scale=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress_bar.close()
        self.file.close()

    def read(self, size):
        data = self.file.read(size)
        self.progress_bar.update(len(data))
        return data

    def readline(self):
        data = self.file.readline()
        self.progress_bar.update(len(data))
        return data


def _tokenize_fn(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int = 2048,
) -> Dict:
    """Tokenize a list of strings."""
    print("+ tokenizing the strings")
    start = time.time()
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )
        for text in tqdm(strings)
    ]
    print(f"  - tokenized {len(strings)} strings in {time.time() - start:.2f}s")

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    system_prompt: Sequence[str],
    questions: Sequence[str],
    responses: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int = 2048,
) -> Dict:
    """Preprocess the data by tokenizing."""
    x, y, examples = [], [], []
    examples = list()
    for sys, question, response in zip(system_prompt, questions, responses):
        sys_prompt_formatted = f"<<SYS>>\n{sys}\n<</SYS>>\n\n"
        question_formatted = f"[INST] {question} [/INST]\n"
        response_formatted = f"{response}\n"
        x += [sys_prompt_formatted + question_formatted]
        y += [response_formatted]
        examples += [sys_prompt_formatted + question_formatted + response_formatted]

    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer, max_length) for strings in (examples, x)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, datasets, tokenizer, use_local_cache: bool = True):
        super(SupervisedDataset, self).__init__()

        self.input_ids = []
        self.labels = []

        # preprocess all data
        index = 0

        if not use_local_cache:
            # split by role
            system_promtps, questions, responses = [], [], []
            print(f"+ preprocessing {len(datasets['train'])} train data")
            for item in tqdm(datasets["train"]):
                if item.get("conversations") is not None:
                    # for slimorca dataset
                    new_item = dict()
                    if item["conversations"][0]["from"] == "system":
                        new_item["system_prompt"] = item["conversations"][0]["value"]
                        new_item["question"] = item["conversations"][1]["value"]
                        new_item["response"] = item["conversations"][2]["value"]
                    elif item["conversations"][0]["from"] == "human":
                        new_item["system_prompt"] = DEFAULT_SYSTEM_PROMPT
                        new_item["question"] = item["conversations"][0]["value"]
                        new_item["response"] = item["conversations"][1]["value"]
                    item = new_item
                _sys_prompt = item.get("system_prompt", "")
                if _sys_prompt == "":
                    _sys_prompt = DEFAULT_SYSTEM_PROMPT

                system_promtps.append(_sys_prompt)
                questions.append(self.clean_question(item["question"]))
                responses.append(item["response"])

            self.train_data = preprocess(
                system_promtps, questions, responses, tokenizer
            )

            joblib.dump(self.train_data, "./.cache/cached_dataset.pkl")
            print(f"  - saved to ./cache/cached_dataset.pkl")
        else:
            print(f"+ loading from ./cache/cached_dataset.pkl")
            start = time.time()
            with ProgressBarFile("./.cache/cached_dataset.pkl") as f:
                self.train_data = cPickle.load(f)

        # check the data
        """ids_check = self.train_data["input_ids"][11]
        labels_check = self.train_data["labels"][11]
        check_str = ""
        for i, (id_, label) in enumerate(zip(ids_check, labels_check)):
            if label == IGNORE_INDEX:
                # print in red
                check_str += f" \033[91m{tokenizer.decode(id_)}\033[0m"
            else:
                # print in green
                check_str += f" \033[92m{tokenizer.decode(id_)}\033[0m"
        print(f"{check_str}")
        exit()"""

    def clean_question(self, question: str):
        if question.startswith("Q:"):
            question = question[2:]
        if question.endswith("A:"):
            question = question[:-2]
        if question.endswith("Output:"):
            question = question[:-7]
        if question.endswith("Output: "):
            question = question[:-8]
        return question

    def __len__(self):
        return len(self.train_data["input_ids"])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.train_data["input_ids"][i],
            labels=self.train_data["labels"][i],
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )

        # Calculate the maximum length for padding
        input_max_length = max(len(input_id) for input_id in input_ids)

        # Left pad the input_ids
        input_ids_padded = []
        for input_id in input_ids:
            # Flip the sequence, pad it, and then flip it back
            input_id_padded = torch.nn.functional.pad(
                input_id.flip(dims=[0]),
                (0, input_max_length - len(input_id)),
                value=self.tokenizer.pad_token_id,
            ).flip(dims=[0])
            input_ids_padded.append(input_id_padded)

        input_ids = torch.stack(input_ids_padded)

        # Left pad the labels
        labels_padded = []
        for label in labels:
            label_padded = torch.nn.functional.pad(
                label.flip(dims=[0]),
                (0, input_max_length - len(label)),
                value=IGNORE_INDEX,
            ).flip(dims=[0])
            labels_padded.append(label_padded)

        labels = torch.stack(labels_padded)

        # for synthetic data
        xs = []

        for i in range(labels.size(0)):
            # Use bitwise '&' for 'and' and ensure each condition is enclosed in parentheses
            valid_indices = (labels[i] == IGNORE_INDEX) & (
                labels[i] != self.tokenizer.pad_token_id
            )
            valid_indices = valid_indices.nonzero(as_tuple=True)[0]
            x = input_ids[i][valid_indices]
            xs.append(x)
        x_max_length = max(len(x) for x in xs)

        # Left pad the sequences (Left pad for generation)
        xs_padded = []
        for x in xs:
            # Flip the sequence, pad it, and then flip it back
            x_padded = torch.nn.functional.pad(
                x.flip(dims=[0]),
                (0, x_max_length - len(x)),
                value=self.tokenizer.pad_token_id,
            ).flip(dims=[0])
            xs_padded.append(x_padded)

        return dict(
            input_ids=input_ids,
            labels=labels,
            x=torch.stack(xs_padded),
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
