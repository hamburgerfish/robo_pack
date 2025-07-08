import pytest
import torch
import tempfile
import os
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from robo_lib import RoboConstructor, save_component, load_component

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- FIXTURES AND MOCKS ----------

@pytest.fixture
def mock_encoder_block():
    return MagicMock()

@pytest.fixture
def mock_decoder_block():
    return MagicMock()

@pytest.fixture
def mock_my_sequential():
    class DummySequential(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, *args, **kwargs):
            return args[0], None, None, None, None
    return DummySequential

@pytest.fixture
def dummy_tokenizer():
    return SimpleNamespace(
        start_token=1,
        end_token=2,
        pad_token=0,
        new_line_token=3,
        encode=lambda s: [4, 5, 6],
        decode=lambda tokens: "decoded"
    )

@pytest.fixture
def dummy_data():
    return torch.randint(0, 10, (8, 32)).to(device)  # 8 samples of 32 tokens

@pytest.fixture
def robo_decoder_only(mock_my_sequential):
    with patch("robo_lib.DecoderBlock", return_value=MagicMock()), \
         patch("robo_lib.MySequential", mock_my_sequential):
        return RoboConstructor(
            n_embed=16,
            dec_n_blocks=2,
            dec_n_head=2,
            dec_vocab_size=50,
            dec_block_size=32
        ).to(device)

@pytest.fixture
def robo_enc_dec(mock_my_sequential):
    with patch("robo_lib.DecoderBlock", return_value=MagicMock()), \
         patch("robo_lib.EncoderBlock", return_value=MagicMock()), \
         patch("robo_lib.MySequential", mock_my_sequential):
        return RoboConstructor(
            n_embed=16,
            dec_n_blocks=2,
            dec_n_head=2,
            dec_vocab_size=50,
            dec_block_size=32,
            enc_n_blocks=2,
            enc_n_head=2,
            enc_vocab_size=50,
            enc_block_size=32
        ).to(device)

# ---------- TESTS ----------

def test_decoder_only_init(robo_decoder_only):
    assert not robo_decoder_only.cross_attention
    assert robo_decoder_only.decoder_blocks is not None
    assert robo_decoder_only.encoder_blocks is None

def test_encoder_decoder_init(robo_enc_dec):
    assert robo_enc_dec.cross_attention
    assert robo_enc_dec.encoder_blocks is not None

def test_forward_decoder_only(robo_decoder_only):
    input_tensor = torch.randint(0, 50, (2, 32)).to(device)
    output = robo_decoder_only(dec_in=input_tensor)
    assert output.shape[:2] == (2, 32)

def test_forward_encoder_decoder(robo_enc_dec):
    dec_input = torch.randint(0, 50, (2, 32)).to(device)
    enc_input = torch.randint(0, 50, (2, 32)).to(device)
    output = robo_enc_dec(dec_in=dec_input, enc_in=enc_input)
    assert output.shape[:2] == (2, 32)

@patch("robo_lib.get_batch")
def test_prep_data_decoder_only(mock_get_batch, robo_decoder_only, dummy_data):
    mock_get_batch.return_value = (dummy_data[:2], dummy_data[:2], dummy_data[:2])
    out = robo_decoder_only.prep_data(batch_size=2, dec_data=dummy_data, dec_block_size=32)
    assert len(out) == 5
    assert out[0].shape[0] == 2

@patch("robo_lib.get_batch")
def test_prep_data_encoder_decoder(mock_get_batch, robo_enc_dec, dummy_data):
    mock_get_batch.side_effect = [
        (dummy_data[:2], dummy_data[:2], dummy_data[:2]),  # decoder
        (dummy_data[:2], None, dummy_data[:2])             # encoder
    ]
    out = robo_enc_dec.prep_data(batch_size=2, dec_data=dummy_data, dec_block_size=32, enc_data=dummy_data, enc_block_size=32)
    assert len(out) == 5
    assert out[3].shape[0] == 2  # encoder input

@patch("robo_lib.top_kp_filter", return_value=torch.tensor([2]))
def test_generate_decoder_only(mock_top_kp, robo_decoder_only, dummy_tokenizer):
    out = robo_decoder_only.generate(inputs="hello", dec_tokenizer=dummy_tokenizer, max_new_tokens=3, dec_start_token=1, dec_end_token=2)
    assert isinstance(out, str)

@patch("robo_lib.top_kp_filter", return_value=torch.tensor([2]))
def test_generate_encoder_decoder(mock_top_kp, robo_enc_dec, dummy_tokenizer):
    out = robo_enc_dec.generate(inputs="hello", enc_tokenizer=dummy_tokenizer, dec_tokenizer=dummy_tokenizer,
                                max_new_tokens=3, enc_start_token=1, enc_end_token=2, dec_start_token=1, dec_end_token=2)
    assert isinstance(out, str)

def test_save_and_load_component(robo_decoder_only):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model")
        save_component(robo_decoder_only, path)
        loaded = load_component(path)
        assert isinstance(loaded, RoboConstructor)
