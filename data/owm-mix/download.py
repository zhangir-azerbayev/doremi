from pathlib import Path
import json
from functools import partial
from itertools import chain
import argparse

from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict

import tensorflow as tf
import sentencepiece as spm

def tokenize(batch, sp_tokenizer, max_length, domain_id):
    # tokenize
    examples = {"input_ids": sp_tokenizer.encode_as_ids(batch['text'])}
    examples["attention_mask"] = [[True for _ in x] for x in examples["input_ids"]]

    # Concatenate all texts.
    examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in examples.items()
    }
    result["domain_id"] = [domain_id for _ in range(len(result['input_ids']))]
    return result

def filter_arxiv(example):
    meta = json.loads(example["meta"])
    return "config" in meta and meta["config"]=="arxiv"

def filter_stack_exchange(example):
    meta = json.loads(example["meta"])
    return "set_name" in meta and meta["set_name"]=="stack_exchange"

def filter_rest(example):
    meta = json.loads(example["meta"])
    return not any((
            ("set_name" in meta and meta["set_name"]=="MATH"),
            filter_arxiv(example),
            filter_stack_exchange(example),
    ))

filter_dict = {
        "arxiv": filter_arxiv,
        "stack_exchange": filter_stack_exchange,
        "rest": filter_rest,
}

def trim_dataset(ds, max_rows: int):
    return DatasetDict({
        k: x.select(range(min(max_rows, len(x)))) 
        for k,x in ds.items()
    })


def main(args):
    max_rows = args.max_rows

    proto = tf.io.gfile.GFile('/home/za2514/downloaded-weights/llama/tokenizer.model', 'rb').read()
    sp = spm.SentencePieceProcessor()
    sp.load_from_serialized_proto(proto)

    for ds_name in args.dataset:
        if ds_name=='proof-pile':
            print("downloading proof-pile...")
            ds = load_dataset("hoskinson-center/proof-pile")

            for domain_id, subset in enumerate(("arxiv", "stack_exchange", "rest")):
                print(f"creating, tokenizing, saving {subset}")

                flter = filter_dict[subset]

                ds_subset = ds.filter(flter)
                
                if max_rows:
                    ds_subset = trim_dataset(ds_subset, max_rows)
                ds_toks = ds_subset.map(
                        partial(tokenize, sp_tokenizer=sp, max_length=args.max_length, domain_id=domain_id),
                        batched=True,
                        remove_columns=['text', 'meta'],
                        num_proc=args.num_proc,
                )
 
                for split in ["train", "test", "validation"]:
                    ds_toks[split].save_to_disk(f"{split}/{subset}")

        elif ds_name=='open-web-math':
            ds = load_dataset("keirp/open-web-math-dev")["train"]
            
            # Make train test validation split
            ds = ds.train_test_split(test_size=args.eval_ratio, shuffle=True)
            test_len = len(ds["test"])//2
            ds["validation"], ds["test"] = (
                    ds["test"].select(range(test_len//2)), 
                    ds["test"].select(range(test_len//2, test_len)),
            )

            if max_rows:
                ds= trim_dataset(ds, max_rows)

            ds_toks = ds.map(
                    partial(tokenize, sp_tokenizer=sp, max_length=args.max_length, domain_id=3),
                    batched=True,
                    remove_columns=['url', 'text', 'metadata'],
                    num_proc=args.num_proc,
            )

            
            for split in ["train", "test", "validation"]:
                ds_toks[split].save_to_disk(f"{split}/open-web-math")
        else:
            raise ValueError("invalid dataset name {ds_name}")

                
if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--dataset", type=str, nargs='+', 
            default=['proof-pile', 'open-web-math']
    )
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--eval_ratio", type=float, default=0.01)

    args = parser.parse_args()

    main(args)
