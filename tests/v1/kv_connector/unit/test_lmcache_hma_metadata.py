# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
    ReqMeta,
    RequestTracker,
)

pytestmark = pytest.mark.cpu_test


def _make_group(block_size: int, layer_name: str):
    return SimpleNamespace(
        kv_cache_spec=SimpleNamespace(block_size=block_size),
        layer_names=[layer_name],
    )


def test_req_meta_uses_group_block_sizes_for_slot_mappings():
    tracker = RequestTracker(
        req_id="req",
        prompt_len=6805,
        token_ids=list(range(6805)),
        allocated_block_ids_by_group=(
            list(range(0, 213)),
            list(range(213, 426)),
            list(range(426, 639)),
            list(range(639, 852)),
            list(range(852, 1065)),
            list(range(1065, 1491)),
        ),
    )
    kv_cache_groups = [_make_group(32, f"l{i}") for i in range(5)] + [
        _make_group(16, "l5")
    ]

    req = ReqMeta.from_request_tracker(
        tracker,
        block_size=16,
        kv_cache_groups=kv_cache_groups,
        lmcache_chunk_size=64,
        discard_partial_chunks=False,
    )

    assert req is not None
    assert req.slot_mapping.shape[0] == 6805
    assert req.block_ids_by_group == tracker.allocated_block_ids_by_group
    assert req.slot_mappings_by_group is not None
    assert [mapping.shape[0] for mapping in req.slot_mappings_by_group] == [6805] * 6
    assert req.slot_mappings_by_layer is not None
    assert all(
        mapping.shape[0] == 6805
        for mapping in req.slot_mappings_by_layer.values()
    )


def test_req_meta_treats_null_blocks_as_pad_slots():
    tracker = RequestTracker(
        req_id="req",
        prompt_len=100,
        token_ids=list(range(100)),
        allocated_block_ids_by_group=([0, 0, 5, 6],),
    )

    req = ReqMeta.from_request_tracker(
        tracker,
        block_size=32,
        kv_cache_groups=[_make_group(32, "l0")],
        lmcache_chunk_size=64,
        discard_partial_chunks=False,
    )

    assert req is not None
    assert req.slot_mappings_by_group is not None
    slot_mapping = req.slot_mappings_by_group[0]
    assert torch.all(slot_mapping[:64] == -1)
    assert torch.equal(slot_mapping[64:96], torch.arange(160, 192, dtype=torch.long))
    assert torch.equal(slot_mapping[96:100], torch.arange(192, 196, dtype=torch.long))


def test_record_failed_blocks_uses_group_specific_block_sizes():
    kv_cache_groups = [_make_group(32, "l0"), _make_group(16, "l1")]
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    impl._block_size = 16
    impl.kv_cache_config = SimpleNamespace(kv_cache_groups=kv_cache_groups)

    expected_mask = torch.ones(64, dtype=torch.bool)
    ret_mask = torch.ones(64, dtype=torch.bool)
    ret_mask[48:] = False

    group0_slot_mapping = torch.arange(320, 384, dtype=torch.long)
    group1_slot_mapping = torch.arange(320, 384, dtype=torch.long)

    missing_blocks = impl.record_failed_blocks(
        "req",
        expected_mask,
        ret_mask,
        group0_slot_mapping,
        (group0_slot_mapping, group1_slot_mapping),
        ([1, 2], [10, 11, 12, 13]),
    )

    assert missing_blocks == {2, 13}


def test_record_failed_blocks_ignores_null_blocks_in_request_tables():
    kv_cache_groups = [_make_group(32, "l0"), _make_group(16, "l1")]
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    impl._block_size = 16
    impl.kv_cache_config = SimpleNamespace(kv_cache_groups=kv_cache_groups)

    expected_mask = torch.ones(96, dtype=torch.bool)
    ret_mask = torch.ones(96, dtype=torch.bool)
    ret_mask[48:] = False

    group0_slot_mapping = torch.arange(0, 96, dtype=torch.long)
    group1_slot_mapping = torch.arange(0, 96, dtype=torch.long)

    missing_blocks = impl.record_failed_blocks(
        "req",
        expected_mask,
        ret_mask,
        group0_slot_mapping,
        (group0_slot_mapping, group1_slot_mapping),
        ([0, 0, 5], [10, 11, 12, 13, 14, 15]),
    )

    assert missing_blocks == {5, 13, 14, 15}
