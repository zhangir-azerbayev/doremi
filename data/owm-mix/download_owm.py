from datasets import load_dataset
from pathlib import Path
from transformers import AutoTokenizer
from functools import partial 

import sys
sys.path.insert(0, './')

from utils import tokenize

import tensorflow as tf
import sentencepiece as spm

max_rows = 100

proto = tf.io.gfile.GFile('/home/za2514/downloaded-weights/llama/tokenizer.model', 'rb').read()
sp = spm.SentencePieceProcessor()
sp.load_from_serialized_proto(proto)
    
Path("open-web-math").mkdir()

ds = load_dataset("keirp/open-web-math-dev")["train"].select(range(0, max_rows))

ds_toks = ds.map(
        partial(tokenize, sp_tokenizer=sp, max_length=1024),
        batched=True,
        remove_columns=['url', 'text', 'metadata'],
        num_proc=4,
)

ds_toks.save_to_disk("open-web-math")
