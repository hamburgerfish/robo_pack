# robo-lib

provides tools for creating, configuring, and training custom transformer models on any data available to you.

## Main features:
- Customize and train tokenizers using an implementation of the features from the [tokenizers](https://pypi.org/project/tokenizers/#description) library.
- Customize data processor to process data into individual tensors, ready to be used to train transformers without further processing.
- Configure transformer models to fit specific requirements/specifications without having to write the internal logic.
- Use the 3 components to create, train, and use custom transformers in different applications.

## Installation

```bash
pip install robo-lib
```

## using robo-lib

Documentation can be found [here](https://github.com/hamburgerfish/robo_pack/wiki).

### Language translation example
- In this example, an encoder-decoder transformer is created for language translation, from English to French.
- This example uses two .txt files for training, one with English, and the other with the equivalent French sentence in each line (delimited by "\n").
- Create, train, and save tokenizers using `TokenizerConstructor`.
- In this example, the WordLevel tokenizer is used, along with the detault arguments of `TokenizerConstructor`.

```python
import robo_lib as rl

encoder_tok = rl.TokenizerConstructor(tokenizer_type="WordLevel")
encoder_tok.train("english_data.txt")

decoder_tok = rl.TokenizerConstructor(tokenizer_type="WordLevel")
encoder_tok.train("french_data.txt")

rl.save_component(encoder_tok, "tokenizers/encoder_tok.pkl")
rl.save_component(decoder_tok, "tokenizers/decoder_tok.pkl")
```

- The `DataProcessor` can be used to automatically process the data into a single torch.tensor, easily useable by the transformer for training.
- The tokenizer(s) must be specified when initialising a DataProcessor. In this case the dec_tokenizer, and enc_tokenizer is both specified for an encoder-decoder transformer.
- The `process_list` method processes lists of string data, so our .txt files are read into lists to be processed by `process_list`.
- In this example, we are splitting the data 90% : 10% for training and validation.

```python
proc = rl.DataProcessor(dec_tokenizer=decoder_tok, enc_tokenizer=encoder_tok)

# read training .txt files into lists
with open("english_data.txt", "r") as file:
    english_list = file.read().split("\n")

with open("french_data.txt", "r") as file:
    french_list = file.read().split("\n")

# splitting lists into train and validation sets
split = 0.9
n = int(len(english_list) * 0.9)
english_train = english_list[:n]
french_train = french_list[:n]
english_val = english_list[n:]
french_val = french_list[n:]

# process and save training data as data/training*.pt
# block_size_exceeded_policy="skip" removes training data larger than specified block size
proc.process_list(
    save_path="data/training",
    dec_data=french_train,
    dec_max_block_size=100,
    dec_block_size_exceeded_policy="skip",
    enc_data=english_train,
    enc_max_block_size=100,
    enc_block_size_exceeded_policy="skip"
)

# process and save validation data as data/validation*.pt
proc.process_list(
    save_path="data/validation",
    dec_data=french_val,
    dec_max_block_size=100,
    dec_block_size_exceeded_policy="skip",
    enc_data=english_val,
    enc_max_block_size=100,
    enc_block_size_exceeded_policy="skip"
)
```
- The `RoboConstructor` class is used to create and configure transformer models before trainin.
- A separate .py file is recommended for training.
- If device is not specified, `RoboConstructor` will take the first available one out of ("cuda", "mps", "cpu"). Torch cuda is not part of the dependencies when installing robo-lib, so it is highly recommended to install it, using this [link](https://pytorch.org/get-started/locally/), if you have a CUDA compatible device.
- The `train` method is used to train the transformer and save it to `save_path` every `eval_interval` iterations.
- If a non-`TokenizerConstructor` token is used, the pad token if your tokenizer can be specified instead of the dec_tokenizer parameter.

```python
import robo_lib as rl

encoder_tok = rl.load_component("tokenizers/encoder_tok.pkl")
decoder_tok = rl.load_component("tokenizers/decoder_tok.pkl")

robo = rl.RoboConstructor(
    n_embed=512,
    dec_n_blocks=6,
    dec_n_head=8,
    dec_vocab_size=decoder_tok.vocab_size,
    dec_block_size=100,
    enc_n_blocks=6,
    enc_n_head=8,
    enc_vocab_size=encoder_tok.vocab_size,
    enc_block_size=100
)

robo.train(
    max_iters=20000,
    eval_interval=200,
    batch_size=128,
    dec_training_path="data/training_decoder_data.pt",
    dec_eval_path="data/validation_decoder_data.pt",
    dec_training_masks_path="data/training_decoder_mask_data.pt",
    dec_eval_masks_path="data/validation_decoder_mask_data.pt",
    enc_training_path="data/training_encoder_data.pt",
    enc_eval_path="data/validation_encoder_data.pt",
    enc_training_masks_path="data/training_encoder_mask_data.pt",
    enc_eval_masks_path="data/validation_encoder_mask_data.pt",
    dec_tokenizer=decoder_tok,
    save_path="models/eng_to_fr_robo.pkl"
)
```

- For language translation, a loss of around 3 already shows good results.
- To use the trained transformer, the `generate` method can be employed.
- The temperature, top_k and top_p values can be specified for this method, along with the tokenizers used.
- If a non-`TokenizerConstructor` tokenizer is used, the start, end, separator (decoder-only), and new-line tokens can be specified of your tokenizer.
- In this example, a simple script is created to interact with the user on the command-line, where the user's English input will be translated by the transformer and printed out onto the console in French.

```python
import robo-lib as rl

robo = rc.load_component("models/eng_to_fr_robo.pkl")
encoder_tok = rl.load_component("tokenizers/encoder_tok.pkl")
decoder_tok = rl.load_component("tokenizers/decoder_tok.pkl")

While True:
    query = input()
    print(robo.generate(query, dec_tokenizer=decoder_tok, enc_tokenizer=encoder_tok))
```

### Shakespeare dialogue generator example
- In this example, a decoder-only transformer is created and trained on a file containing all the dialogue written by William Shakespeare in his plays.
- The training data is in the form of a single .txt file containing the dialogue.
- The default BPE tokenizer is used in this case, so no argument is specified for `TokenizerConstructor`.

```python
import robo_lib as rl

tok = rl.TokenizerConstructor()
tok.train("shakespeare_dialogues.txt")

rl.save_component(tok, "tokenizers/shakespeare_tok.pkl")
```

- In this example, instead of having multiple pieces of training data, we have one large text file, from which random chunks of length `block_size` can be used for training. Therefore, a single large string is input into the DataProcessor instead of a list of strings.
- Since this is a decoder-only transformer, encoder arguments are not given.
- Since the entire string should be processed as is, instead of creating blocks of training data, block_size is not specified.
- dec_create_masks is set to False, as there will be no padding in the training data.

```python
proc = rl.DataProcessor(dec_tokenizer=tok)

# read training .txt file
with open("shakespeare_dialogues.txt", "r") as file:
    dialogues_str = file.read()

# splitting string into train and validation sets
split = 0.9
n = int(len(dialogues_str) * 0.9)
train_data = dialogues_str[:n]
val_data = dialogues_str[n:]

# process and save training data as data/shakespeare_train*.pt
proc.process_list(
    save_path="data/shakespeare_train",
    dec_data=train_data,
    dec_create_masks=False
    )

# process and save validation data as data/validation*.pt
proc.process_list(
    save_path="data/shakespeare_valid",
    dec_data=val_data,
    dec_create_masks=False
)
```
- Training the transformer.
```python
import robo_lib as rl

tok = rl.load_component("tokenizers/shakespeare_tok.pkl")

robo = rl.RoboConstructor(
    n_embed=1024,
    dec_n_blocks=8,
    dec_n_head=8,
    dec_vocab_size=tok.vocab_size,
    dec_block_size=200
)

robo.train(
    max_iters=20000,
    eval_interval=200,
    batch_size=64,
    dec_training_path="data/shakespeare_train_decoder_data.pt",
    dec_eval_path="data/shakespeare_valid_decoder_data.pt",
    dec_tokenizer=tok,
    save_path="models/shakespeare_robo.pkl"
)
```
- In this example, the user can specify the start of the generated Shakespeare play and the transformer will generate and print the rest, until `max_new_tokens` (1000) tokens are generated.
- Temperature and top_k are set to 1.2 and 2 respectively to generate a more "creative" output.
```python
import robo_lib as rl

robo = rc.load_component("models/shakespeare_robo.pkl")
tok = rl.load_component("tokenizers/shakespeare_tok.pkl")

While True:
    start = input()
    print(robo.generate(start, max_new_tokens=1000, dec_tokenizer=tok, temperature=1.2, top_k=2))
```