from datasets import load_dataset 
from pathlib import Path
import json
from functools import partial

from transformers import AutoTokenizer

import sys
sys.path.insert(0, './')

from utils import tokenize

import tensorflow as tf
import sentencepiece as spm

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

max_rows = 100

proto = tf.io.gfile.GFile('/home/za2514/downloaded-weights/llama/tokenizer.model', 'rb').read()
sp = spm.SentencePieceProcessor()
sp.load_from_serialized_proto(proto)

print("downloading proof-pile...")
ds = load_dataset("hoskinson-center/proof-pile")["train"]

print("creating and saving arxiv...")

Path("arxiv").mkdir()
ds_arxiv = ds.filter(filter_arxiv).select(range(0, max_rows))
ds_arxiv = ds_arxiv.map(
        partial(tokenize, sp_tokenizer=sp, max_length=1024),
        batched=True,
        remove_columns=['text', 'meta'],
        num_proc=4,
)
ds_arxiv.save_to_disk("arxiv")

print("creating and saving stack exchange")

Path("stack_exchange").mkdir()
ds_stack = ds.filter(filter_stack_exchange).select(range(0, max_rows))
ds_stack = ds_stack.map(
        partial(tokenize, sp_tokenizer=sp, max_length=1024),
        batched=True,
        remove_columns=['text', 'meta'],
        num_proc=4,
)
ds_stack.save_to_disk("stack_exchange")

print("creating and saving rest of proof-pile...")
Path("rest").mkdir()
ds_rest = ds.filter(filter_rest).select(range(0, max_rows))
ds_rest = ds_rest.map(
        partial(tokenize, sp_tokenizer=sp, max_length=1024),
        batched=True,
        remove_columns=['text', 'meta'],
        num_proc=4,
)
ds_rest.save_to_disk("rest")
