from itertools import chain

def tokenize(batch, sp_tokenizer, max_length):
    # tokenize
    examples = {"input_ids": sp_tokenizer.encode_as_ids(batch['text'])}

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
    return result
