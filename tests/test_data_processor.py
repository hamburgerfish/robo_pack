import os
import shutil
import torch
import pytest
from robo_lib.components import DataProcessor, TokenizerConstructor


@pytest.fixture
def temp_save_path():
    path = "temp_test_dir"
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)


@pytest.fixture
def dummy_tokenizer():
    tokenizer = TokenizerConstructor(
        tokenizer_type="WordLevel",
        pre_tokenizers="Whitespace",
        special_tokens=["<pad>", "<unk>"],
        vocab={"hello": 0, "world": 1, "<pad>": 2, "<unk>": 3},
        unknown_token_string="<unk>",
        pad_token_string="<pad>",
        start_token_string=None,
        end_token_string=None,
        new_line_token_string=None
    )
    # Faking "training" so encode works with the vocab
    tokenizer.tokenizer_type.add_tokens(["hello", "world"])
    return tokenizer


def test_data_processor_initialization(dummy_tokenizer):
    processor = DataProcessor(dec_tokenizer=dummy_tokenizer)
    assert processor.dec_tokenizer is dummy_tokenizer
    assert processor.enc_tokenizer is None


def test_process_list_decoder_only(dummy_tokenizer, temp_save_path):
    processor = DataProcessor(dec_tokenizer=dummy_tokenizer)
    data = ["hello world", "world hello"]
    
    processor.process_list(
        dec_data=data,
        dec_max_block_size=10,
        save_path=temp_save_path
    )

    assert os.path.exists(os.path.join(temp_save_path, "decoder_data.pt"))
    assert os.path.exists(os.path.join(temp_save_path, "decoder_mask_data.pt"))

    tensor = torch.load(os.path.join(temp_save_path, "decoder_data.pt"))
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[0] == len(data)


def test_process_list_encoder_decoder(dummy_tokenizer, temp_save_path):
    processor = DataProcessor(dec_tokenizer=dummy_tokenizer, enc_tokenizer=dummy_tokenizer)
    data = ["hello world", "world hello"]

    processor.process_list(
        dec_data=data,
        enc_data=data,
        dec_max_block_size=10,
        enc_max_block_size=10,
        save_path=temp_save_path
    )

    assert os.path.exists(os.path.join(temp_save_path, "decoder_data.pt"))
    assert os.path.exists(os.path.join(temp_save_path, "encoder_data.pt"))
    assert os.path.exists(os.path.join(temp_save_path, "decoder_mask_data.pt"))
    assert os.path.exists(os.path.join(temp_save_path, "encoder_mask_data.pt"))


def test_process_list_mismatched_lengths_raises(dummy_tokenizer):
    processor = DataProcessor(dec_tokenizer=dummy_tokenizer)
    dec_data = ["hello world"]
    enc_data = ["world hello", "extra row"]

    with pytest.raises(Exception, match="decoder data and encoder data lengths do not match"):
        processor.process_list(dec_data=dec_data, enc_data=enc_data)
