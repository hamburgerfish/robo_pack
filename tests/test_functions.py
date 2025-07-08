import pytest
import torch
import numpy as np
import random
from robo_lib import create_mask, pre_process_data, safe_stack, get_valid_samples, get_batch, top_kp_filter

def test_create_mask_basic():
    row = [1, 2, 3]
    block_size = 5
    expected = [1, 1, 1, 0, 0]
    assert create_mask(row, block_size) == expected

def test_create_mask_equal_length():
    row = [1, 2, 3, 4]
    block_size = 4
    expected = [1, 1, 1, 1]
    assert create_mask(row, block_size) == expected

def test_create_mask_empty_row():
    row = []
    block_size = 3
    expected = [0, 0, 0]
    assert create_mask(row, block_size) == expected

def test_pre_process_data_none_tokens():
    data = ["hello", "world"]
    start_token = None
    end_token = None
    # Should return the input unchanged
    assert pre_process_data(data.copy(), start_token, end_token) == data

def test_pre_process_data_start_token_only():
    data = ["hello", "world"]
    start_token = "<s>"
    end_token = None
    expected = ["<s>hello", "<s>world"]
    assert pre_process_data(data.copy(), start_token, end_token) == expected

def test_pre_process_data_end_token_only():
    data = ["hello", "world"]
    start_token = None
    end_token = "</s>"
    expected = ["hello</s>", "world</s>"]
    assert pre_process_data(data.copy(), start_token, end_token) == expected

def test_pre_process_data_both_tokens():
    data = ["hello", "world"]
    start_token = "<s>"
    end_token = "</s>"
    expected = ["<s>hello</s>", "<s>world</s>"]
    assert pre_process_data(data.copy(), start_token, end_token) == expected

def test_safe_stack_valid_tensors():
    t1 = torch.tensor([1, 2])
    t2 = torch.tensor([3, 4])
    tensor_list = [t1, t2]
    stacked = safe_stack(tensor_list)
    assert isinstance(stacked, torch.Tensor)
    assert stacked.shape == (2, 2)

def test_safe_stack_ignore_non_tensors():
    t1 = torch.tensor([1, 2])
    not_tensor = [1, 2, 3]
    tensor_list = [t1, not_tensor]
    stacked = safe_stack(tensor_list)
    assert stacked.shape == (1, 2)

def test_safe_stack_raises_for_empty():
    with pytest.raises(ValueError):
        safe_stack(["not a tensor", 123, None])


# For reproducibility
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

def test_get_valid_samples_all_masked_less_than_block():
    masks = torch.tensor([[1, 0, 0], [1, 1, 0]])
    random_samples = torch.tensor([0, 1])
    block_size = 2
    result = get_valid_samples(random_samples, masks, block_size)
    # For first row sum(masks) = 1 <= block_size => should return 0
    # For second row sum(masks) = 2 <= block_size => 0
    assert result == [0, 0]

def test_get_valid_samples_some_greater_than_block():
    masks = torch.tensor([[1, 1, 1], [1, 1, 0]])
    random_samples = torch.tensor([0, 1])
    block_size = 2
    result = get_valid_samples(random_samples, masks, block_size)
    # first sum = 3 > 2, so random index in [0, 1]
    # second sum = 2 <= 2, so 0
    assert result[1] == 0
    assert 0 <= result[0] <= 1

def test_get_batch_no_masks_get_offset_true():
    data = torch.arange(30).view(5, 6)  # 5 rows, 6 cols
    random_samples = torch.tensor([0, 1, 2])
    block_size = 4
    batch_in, batch_out, masks_in = get_batch(data, random_samples, masks=None, block_size=block_size, get_offset=True)
    assert batch_in.shape == (3, block_size-1)
    assert batch_out.shape == (3, block_size-1)
    assert masks_in is None

def test_get_batch_with_masks_get_offset_false():
    data = torch.arange(30).view(5, 6)
    masks = torch.ones_like(data)
    random_samples = torch.tensor([0, 1])
    block_size = 5
    batch_in, batch_out, masks_in = get_batch(data, random_samples, masks=masks, block_size=block_size, get_offset=False)
    assert batch_in.shape == (2, block_size)
    assert batch_out is None
    assert masks_in.shape == (2, block_size)

def test_get_batch_block_size_larger_than_data_length_raises():
    data = torch.arange(20).view(4, 5)
    random_samples = torch.tensor([0])
    block_size = 6
    with pytest.raises(Exception):
        get_batch(data, random_samples, block_size=block_size)


def test_top_kp_filter_top_k_only():
    # Create dummy logits batch (2 samples, vocab size 5)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                           [5.0, 4.0, 3.0, 2.0, 1.0]])
    top_k = 3
    selected = top_kp_filter(logits, top_k=top_k, top_p=None)

    assert selected.shape == (2,)
    # Selected indices must be in top_k tokens
    for i, sel in enumerate(selected):
        topk_indices = torch.topk(logits[i], top_k).indices.tolist()
        assert sel.item() in topk_indices

def test_top_kp_filter_top_p_only():
    # Dummy logits with clear probabilities
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                           [0.4, 0.3, 0.2, 0.1]])
    top_p = 0.7
    selected = top_kp_filter(logits, top_k=None, top_p=top_p)

    assert selected.shape == (2,)
    # Selected indices must be in vocab range
    for sel in selected:
        assert 0 <= sel.item() < logits.shape[1]

def test_top_kp_filter_top_k_and_top_p():
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                           [0.5, 0.4, 0.3, 0.2, 0.1]])
    top_k = 2
    top_p = 0.6
    selected = top_kp_filter(logits, top_k=top_k, top_p=top_p)

    assert selected.shape == (2,)
    for i, sel in enumerate(selected):
        # With both filters, selected index should be in top_k indices
        topk_indices = torch.topk(logits[i], top_k).indices.tolist()
        assert sel.item() in topk_indices

def test_top_kp_filter_no_filter():
    logits = torch.tensor([[0.1, 0.2, 0.3],
                           [0.3, 0.2, 0.1]])
    selected = top_kp_filter(logits, top_k=None, top_p=None)

    assert selected.shape == (2,)
    for sel in selected:
        assert 0 <= sel.item() < logits.shape[1]

def test_top_kp_filter_empty_logits():
    # Edge case: logits empty or zero size
    logits = torch.empty((0, 0))
    with pytest.raises(IndexError):
        _ = top_kp_filter(logits, top_k=1, top_p=0.5)

