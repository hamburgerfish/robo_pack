import pytest
import os
from tempfile import NamedTemporaryFile
from robo_lib import TokenizerConstructor


@pytest.fixture
def training_file():
    with NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write("Hello world\nThis is a test\nTokenizer test\n")
        f.flush()
        yield f.name
    os.remove(f.name)


def test_tokenizer_creation():
    tokenizer = TokenizerConstructor(
        tokenizer_type="BPE",
        pre_tokenizers="Whitespace",
        normalizers=["Lowercase"],
        special_tokens=["<unk>", "<pad>"],
        vocab_size=100
    )
    assert tokenizer is not None
    assert "<unk>" in tokenizer.special_tokens
    assert tokenizer.vocab_size is None  # Untrained tokenizer should have vocab_size None


def test_tokenizer_train(training_file):
    tokenizer = TokenizerConstructor(
        tokenizer_type="WordLevel",
        pre_tokenizers="Whitespace",
        normalizers=["Lowercase"],
        special_tokens=["<unk>", "<pad>"],
        vocab_size=50
    )
    tokenizer.train(training_file)
    assert tokenizer.vocab_size is not None
    assert tokenizer.vocab_size > 0


def test_tokenizer_encode_decode(training_file):
    tokenizer = TokenizerConstructor(
        tokenizer_type="BPE",
        pre_tokenizers="Whitespace",
        normalizers=["Lowercase"],
        special_tokens=["<unk>", "<pad>"],
        vocab_size=50
    )
    tokenizer.train(training_file)
    encoded = tokenizer.encode("This is a test")
    assert isinstance(encoded, list)
    assert all(isinstance(i, int) for i in encoded)

    decoded = tokenizer.decode(encoded)
    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_tokenizer_encode_batch(training_file):
    tokenizer = TokenizerConstructor(
        tokenizer_type="BPE",
        pre_tokenizers="Whitespace",
        normalizers=["Lowercase"],
        special_tokens=["<unk>", "<pad>"],
        vocab_size=50
    )
    tokenizer.train(training_file)
    batch = ["This is a test", "Hello world"]
    encoded_batch = tokenizer.encode_batch(batch)
    assert isinstance(encoded_batch, list)
    assert len(encoded_batch) == len(batch)
    assert all(isinstance(seq, list) for seq in encoded_batch)

    encoded_truncated = tokenizer.encode_batch(batch, max_length=3)
    assert all(len(seq) <= 3 for seq in encoded_truncated)


def test_special_token_indexes():
    tokenizer = TokenizerConstructor(
        tokenizer_type="BPE",
        pre_tokenizers="Whitespace",
        special_tokens=["<unk>", "<sos>", "<eos>", "<pad>", "\n"]
    )
    assert tokenizer.unknown_token == tokenizer.special_tokens.index("<unk>")
    assert tokenizer.start_token == tokenizer.special_tokens.index("<sos>")
    assert tokenizer.end_token == tokenizer.special_tokens.index("<eos>")
    assert tokenizer.pad_token == tokenizer.special_tokens.index("<pad>")
    assert tokenizer.new_line_token == tokenizer.special_tokens.index("\n")
