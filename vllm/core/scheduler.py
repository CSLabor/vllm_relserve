import enum
import os
import random
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple, Union

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceGroupMetadataDelta,
                           SequenceStatus)
from vllm.utils import Device, PyObjectCache

logger = init_logger(__name__)

# Test-only. If configured, decode is preempted with
# ARTIFICIAL_PREEMPTION_PROB% probability.
ENABLE_ARTIFICIAL_PREEMPT = bool(
    os.getenv("VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT", False))  # noqa
ARTIFICIAL_PREEMPTION_PROB = 0.5
ARTIFICIAL_PREEMPTION_MAX_CNT = 500


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


@dataclass
class SchedulingBudget:
    """The available slots for scheduling.

    TODO(sang): Right now, the budget is request_id-aware meaning it can ignore
    budget update from the same request_id. It is because in normal scheduling
    path, we update RUNNING num_seqs ahead of time, meaning it could be
    updated more than once when scheduling RUNNING requests. Since this won't
    happen if we only have chunked prefill scheduling, we can remove this
    feature from the API when chunked prefill is enabled by default.
    """
    token_budget: int
    max_num_seqs: int
    _request_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _request_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    # Number of cached tokens in the batch.
    _num_cached_tokens: int = 0
    # Number of actual non-cached tokens in the batch.
    _num_batched_tokens: int = 0
    _num_curr_seqs: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
        # We allow num_new_tokens to be 0 when the entire sequence has
        # been cached.
        assert num_new_tokens >= 0
        assert num_new_seqs != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)

    def remaining_nseq_budget(self):
        return self.max_num_seqs - self.num_curr_seqs

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self,
                               req_id: str,
                               num_batched_tokens: int,
                               num_cached_tokens: int = 0):
        if req_id in self._request_ids_num_batched_tokens:
            return
        assert num_cached_tokens >= 0
        assert num_batched_tokens >= 0

        self._request_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens
        self._num_cached_tokens += num_cached_tokens

    def subtract_num_batched_tokens(self,
                                    req_id: str,
                                    num_batched_tokens: int,
                                    num_cached_tokens: int = 0):
        if req_id not in self._request_ids_num_batched_tokens:
            return
        assert num_cached_tokens >= 0
        assert num_batched_tokens >= 0

        self._request_ids_num_batched_tokens.remove(req_id)
        self._num_batched_tokens -= num_batched_tokens
        self._num_cached_tokens -= num_cached_tokens

        assert self._num_batched_tokens >= 0
        assert self._num_cached_tokens >= 0

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            return

        self._request_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            self._request_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs

    @property
    def num_cached_tokens(self):
        return self._num_cached_tokens


@dataclass
class ScheduledSequenceGroup:
    # A sequence group that's scheduled.
    seq_group: SequenceGroup
    # The total chunk size (number of tokens) to process for next iteration.
    # 1 for decoding. Same as prompt tokens for prefill, but if prefill is
    # chunked, it can be smaller than that.
    token_chunk_size: int


@dataclass
class SchedulerOutputs:
    """The scheduling decision made from a scheduler."""
    # Scheduled sequence groups.
    scheduled_seq_groups: GenericSequence[ScheduledSequenceGroup]
    # Number of prefill groups scheduled.
    num_prefill_groups: int
    # Total number of batched tokens.
    num_batched_tokens: int
    # Blocks to swap in. List of CPU -> GPU block number.
    blocks_to_swap_in: List[Tuple[int, int]]
    # Blocks to swap out. List of GPU -> CPU block number.
    blocks_to_swap_out: List[Tuple[int, int]]
    # Blocks to copy. Source to dest block.
    blocks_to_copy: List[Tuple[int, int]]
    # Sequence groups that are going to be ignored.
    ignored_seq_groups: List[SequenceGroup]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # The number of requests in the running queue
    running_queue_size: int
    preempted: int
    # Add for profiling
    num_uncached_prefill_tokens: int = -1

    def __post_init__(self):
        # Swap in and swap out should never happen at the same time.
        assert not (self.blocks_to_swap_in and self.blocks_to_swap_out)

        self.num_loras: int = len(self.lora_requests)
        if self.num_loras > 0:
            self._sort_by_lora_ids()

        self.num_prompt_adapters: int = len(self.prompt_adapter_requests)

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)

    def _sort_by_lora_ids(self):
        assert 0 <= self.num_prefill_groups <= len(self.scheduled_seq_groups)

        def key_fn(group: ScheduledSequenceGroup):
            key = (group.seq_group.lora_int_id, group.seq_group.request_id)
            if 0 < self.num_prefill_groups < len(self.scheduled_seq_groups):
                # Sort sequence groups so that all prefills come before all
                # decodes as required by chunked prefill.
                return (not group.seq_group.is_prefill(), *key)
            return key

        self.scheduled_seq_groups = sorted(self.scheduled_seq_groups,
                                           key=key_fn)

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {
            g.seq_group.lora_request
            for g in self.scheduled_seq_groups
            if g.seq_group.lora_request is not None
        }

    @property
    def prompt_adapter_requests(self) -> Set[PromptAdapterRequest]:
        return {
            g.seq_group.prompt_adapter_request
            for g in self.scheduled_seq_groups
            if g.seq_group.prompt_adapter_request is not None
        }


@dataclass
class SchedulerRunningOutputs:
    """The requests that are scheduled from a running queue.

    Could contain prefill (prefill that's chunked) or decodes. If there's not
    enough memory, it can be preempted (for recompute) or swapped out.
    """
    # Selected sequences that are running and in a decoding phase.
    decode_seq_groups: List[ScheduledSequenceGroup]
    # Selected sequences that are running and in a prefill phase.
    # I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[ScheduledSequenceGroup]
    # The preempted sequences.
    preempted: List[SequenceGroup]
    # Sequences that are swapped out.
    swapped_out: List[SequenceGroup]
    # The blocks to swap out.
    blocks_to_swap_out: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int

    # Optimization for fast-access to seq_group lists
    decode_seq_groups_list: List[SequenceGroup]
    prefill_seq_groups_list: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> "SchedulerRunningOutputs":
        return SchedulerRunningOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            preempted=[],
            swapped_out=[],
            blocks_to_swap_out=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
            decode_seq_groups_list=[],
            prefill_seq_groups_list=[],
        )


@dataclass
class SchedulerSwappedInOutputs:
    """The requests that are scheduled from a swap queue.

    Could contain prefill (prefill that's chunked) or decodes.
    """
    # Selected sequences that are going to be swapped in and is in a
    # decoding phase.
    decode_seq_groups: List[ScheduledSequenceGroup]
    # Selected sequences that are going to be swapped in and in a prefill
    # phase. I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[ScheduledSequenceGroup]
    # The blocks to swap in.
    blocks_to_swap_in: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # Infeasible sequence groups.
    infeasible_seq_groups: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> "SchedulerSwappedInOutputs":
        return SchedulerSwappedInOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            blocks_to_swap_in=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
            infeasible_seq_groups=[],
        )


@dataclass
class SchedulerPrefillOutputs:
    """The requests that are scheduled from a waiting queue.

    Could contain a fresh prefill requests or preempted requests that need
    to be recomputed from scratch.
    """
    # Selected sequences for prefill.
    seq_groups: List[ScheduledSequenceGroup]
    # Ignored sequence groups.
    ignored_seq_groups: List[SequenceGroup]
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerPrefillOutputs":
        return SchedulerPrefillOutputs(
            seq_groups=[],
            ignored_seq_groups=[],
            num_lookahead_slots=0,
        )


def seq_group_metadata_builder():
    return SequenceGroupMetadata(request_id="",
                                 is_prompt=False,
                                 seq_data={},
                                 sampling_params=None,
                                 block_tables={})


def scheduler_running_outputs_builder():
    return SchedulerRunningOutputs(decode_seq_groups=[],
                                   prefill_seq_groups=[],
                                   preempted=[],
                                   swapped_out=[],
                                   blocks_to_swap_out=[],
                                   blocks_to_copy=[],
                                   num_lookahead_slots=0,
                                   prefill_seq_groups_list=[],
                                   decode_seq_groups_list=[])


def scheduled_seq_group_builder():
    return ScheduledSequenceGroup(SequenceGroup.__new__(SequenceGroup),
                                  token_chunk_size=0)
    # return ScheduledSequenceGroup(seq_group=None, token_chunk_size=0)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        # Note for LoRA scheduling: the current policy is extremely
        # simple and NOT fair. It can lead to starvation of some
        # LoRAs. This should be improved in the future.
        self.lora_config = lora_config

        version = "selfattn"
        if (self.scheduler_config.runner_type == "pooling"
                or self.cache_config.is_attention_free):
            version = "placeholder"

        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
            version)

        num_gpu_blocks = cache_config.num_gpu_blocks
        if num_gpu_blocks:
            num_gpu_blocks //= pipeline_parallel_size

        num_cpu_blocks = cache_config.num_cpu_blocks
        if num_cpu_blocks:
            num_cpu_blocks //= pipeline_parallel_size

        # Create the block space manager.
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)

        # Sequence groups in the WAITING state.
        # Contain new prefill or preempted requests.
        self.waiting: Deque[SequenceGroup] = deque()
        # Sequence groups in the RUNNING state.
        # Contain decode requests.
        self.running: Deque[SequenceGroup] = deque()
        # Sequence groups in the SWAPPED state.
        # Contain decode requests that are swapped out.
        self.swapped: Deque[SequenceGroup] = deque()
        # Sequence groups finished requests ids since last step iteration.
        # It lets the model know that any state associated with these requests
        # can and must be released after the current step.
        # This is used to evict the finished requests from the Mamba cache.
        self._finished_requests_ids: List[str] = list()
        # Time at previous scheduling step
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0
        # preemption mode, RECOMPUTE or SWAP
        self.user_specified_preemption_mode = scheduler_config.preemption_mode

        # The following field is test-only. It is used to inject artificial
        # preemption.
        self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
        self.artificial_preempt_cnt = (ARTIFICIAL_PREEMPTION_MAX_CNT
                                       if self.enable_artificial_preemption
                                       else 0)
        self.num_cumulative_preemption: int = 0

        # Used to cache python objects
        self._seq_group_metadata_cache: List[PyObjectCache] = []
        self._scheduler_running_outputs_cache: List[PyObjectCache] = []
        self._scheduled_seq_group_cache: List[PyObjectCache] = []

        # For async output processing, we need to swap cache buffers between
        # iterations. I.e. since the output processing is lagged one step,
        # we cannot reuse the cached objects immediately when the schedule()
        # is called again, but only when schedule() is called the second time.
        self.output_proc_callback = output_proc_callback
        self.use_async_output_proc = self.output_proc_callback is not None
        self.num_cache_iters = 2 if self.use_async_output_proc else 1

        self.cache_id = 0
        for i in range(self.num_cache_iters):
            self._seq_group_metadata_cache.append(
                PyObjectCache(seq_group_metadata_builder))
            self._scheduler_running_outputs_cache.append(
                PyObjectCache(scheduler_running_outputs_builder))
            self._scheduled_seq_group_cache.append(
                PyObjectCache(scheduled_seq_group_builder))

        # For async postprocessor, the extra decode run cannot be done
        # when the request reaches max_model_len. In this case, the request
        # will be stopped during schedule() call and added to this stop list
        # for processing and deallocation by the free_finished_seq_groups()
        self._async_stopped: List[SequenceGroup] = []

        self.historical_cache_miss = dict()
        self.historical_staleness = dict()

        self.cached_waiting_priority = None
        self.cached_waiting_queue_status = None

        self.starvation_threshold = dict() # rid --> threshold
        self.relquery_arrival = dict() # rid --> first request arrival time

    @property
    def next_cache_id(self):
        return (self.cache_id + 1) % self.num_cache_iters

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens."""
        return 1

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def _add_seq_group_to_running(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the running queue.
        # Only for testing purposes.
        self.running.append(seq_group)

    def _add_seq_group_to_swapped(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the swapped queue.
        # Only for testing purposes.
        self.swapped.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
        """
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity.
                    break
                if seq_group.request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    request_ids.remove(seq_group.request_id)
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                # Remove the aborted request from the Mamba cache.
                self._finished_requests_ids.append(aborted_group.request_id)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)

                self._free_seq_group_cross_attn_blocks(aborted_group)

    def _free_seq_group_cross_attn_blocks(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        """
        Free a sequence group from a cross-attention block table.
        Has no effect on decoder-only models.
        """
        if seq_group.is_encoder_decoder():
            self.block_manager.free_cross(seq_group)

    def has_unfinished_seqs(self) -> bool:
        return len(self.waiting) != 0 or len(self.running) != 0 or len(
            self.swapped) != 0

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.block_manager.get_prefix_cache_hit_rate(device)

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def get_and_reset_finished_requests_ids(self) -> List[str]:
        """Flushes the list of request ids of previously finished seq_groups."""
        finished_requests_ids = self._finished_requests_ids
        self._finished_requests_ids = list()
        return finished_requests_ids

    def _schedule_running(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerRunningOutputs:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
    
        Returns:
            SchedulerRunningOutputs.
        """
        ret: SchedulerRunningOutputs = \
            self._scheduler_running_outputs_cache[self.cache_id].get_object()
        ret.blocks_to_swap_out.clear()
        ret.blocks_to_copy.clear()
        ret.decode_seq_groups.clear()
        ret.prefill_seq_groups.clear()
        ret.preempted.clear()
        ret.swapped_out.clear()

        ret.num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill=False, enable_chunking=enable_chunking)

        ret.decode_seq_groups_list.clear()
        ret.prefill_seq_groups_list.clear()

        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = ret.blocks_to_swap_out
        blocks_to_copy: List[Tuple[int, int]] = ret.blocks_to_copy

        decode_seq_groups: List[ScheduledSequenceGroup] = ret.decode_seq_groups
        prefill_seq_groups: List[
            ScheduledSequenceGroup] = ret.prefill_seq_groups
        preempted: List[SequenceGroup] = ret.preempted
        swapped_out: List[SequenceGroup] = ret.swapped_out

        running_queue = self.running
        assert len(self._async_stopped) == 0
        while running_queue:
            seq_group = running_queue[0]
            # We discard the cached tokens info here because we don't need it
            # for running sequence:
            #   1. If a sequence is running with chunked prefill, the cached
            #      tokens info was already used for the first prefill.
            #   2. If a sequence is running with non-chunked prefill, then
            #      there it's a decoding sequence, and the cached tokens info is
            #      irrelevant.
            num_uncached_new_tokens, _ = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.RUNNING, enable_chunking,
                    budget))

            num_running_tokens = num_uncached_new_tokens
            if num_running_tokens == 0:
                # No budget => Stop
                break

            running_queue.popleft()

            # With async postprocessor, an extra decode run is done
            # to process the final tokens. The check below avoids this extra
            # decode run when the model max len is reached, in order to avoid
            # a memory overflow.
            if self.use_async_output_proc and seq_group.seqs[0].get_len(
            ) > self.scheduler_config.max_model_len:
                self._async_stopped.append(seq_group)
                continue

            # NOTE(woosuk): Preemption happens only when there is no available
            # slot to keep all the sequence groups in the RUNNING state.
            while not self._can_append_slots(seq_group, enable_chunking):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.remove(seq_group.lora_int_id)

                # Determine victim sequence
                cont_loop = True
                if running_queue:
                    # Preempt the lowest-priority sequence group.
                    victim_seq_group = running_queue.pop()
                else:
                    # No other sequence group can be preempted.
                    # Preempt the current sequence group.
                    # Note: This is also where we stop this loop
                    # (since there is nothing else to preempt)
                    victim_seq_group = seq_group
                    cont_loop = False

                # With async postprocessor, before preempting a sequence
                # we need to ensure it has no pending async postprocessor
                do_preempt = True
                if self.use_async_output_proc:
                    assert self.output_proc_callback is not None
                    self.output_proc_callback(
                        request_id=victim_seq_group.request_id)

                    # It may be that the async pending "victim_seq_group"
                    # becomes finished, in which case we simply free it.
                    if victim_seq_group.is_finished():
                        self._free_finished_seq_group(victim_seq_group)
                        do_preempt = False

                # Do preemption
                if do_preempt:
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)

                if not cont_loop:
                    break
            else:
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                is_prefill = seq_group.is_prefill()

                scheduled_seq_group: ScheduledSequenceGroup = \
                    self._scheduled_seq_group_cache[self.cache_id].get_object()
                scheduled_seq_group.seq_group = seq_group
                if is_prefill:
                    scheduled_seq_group.token_chunk_size = num_running_tokens
                    prefill_seq_groups.append(scheduled_seq_group)
                    ret.prefill_seq_groups_list.append(seq_group)
                else:
                    scheduled_seq_group.token_chunk_size = 1
                    decode_seq_groups.append(scheduled_seq_group)
                    ret.decode_seq_groups_list.append(seq_group)

                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

        self._scheduler_running_outputs_cache[self.next_cache_id].reset()
        self._scheduled_seq_group_cache[self.next_cache_id].reset()

        return ret

    def _schedule_swapped(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerSwappedInOutputs:
        """Schedule sequence groups that are swapped out.

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are swapped in.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are swapped in.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerSwappedInOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        infeasible_seq_groups: List[SequenceGroup] = []

        swapped_queue = self.swapped

        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            seq_group = swapped_queue[0]

            # If the sequence group cannot be swapped in, stop.
            is_prefill = seq_group.is_prefill()
            alloc_status = self.block_manager.can_swap_in(
                seq_group,
                self._get_num_lookahead_slots(is_prefill, enable_chunking))
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the request %s because there's not enough kv "
                    "cache blocks to run the entire sequence.",
                    seq_group.request_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                swapped_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.SWAPPED, enable_chunking,
                    budget))

            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            swapped_queue.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy, enable_chunking)
            is_prefill = seq_group.is_prefill()
            if is_prefill:
                prefill_seq_groups.append(
                    ScheduledSequenceGroup(
                        seq_group,
                        token_chunk_size=num_new_tokens_uncached +
                        num_new_tokens_cached,
                    ))
            else:
                decode_seq_groups.append(
                    ScheduledSequenceGroup(seq_group, token_chunk_size=1))
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        swapped_queue.extendleft(leftover_swapped)

        return SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False, enable_chunking=enable_chunking),
            infeasible_seq_groups=infeasible_seq_groups,
        )

    def _get_prompt_limit(self, seq_group: SequenceGroup) -> int:
        if self.scheduler_config.chunked_prefill_enabled and \
                not self.scheduler_config.is_multi_step:
            prompt_limit = self.scheduler_config.max_model_len
        else:
            prompt_limit = min(self.scheduler_config.max_model_len,
                               self.scheduler_config.max_num_batched_tokens)

        # Model is fine tuned with long context. Return the fine tuned max_len.
        if (seq_group.lora_request
                and seq_group.lora_request.long_lora_max_len):
            assert prompt_limit <= seq_group.lora_request.long_lora_max_len
            return seq_group.lora_request.long_lora_max_len
        else:
            return prompt_limit

    def _get_priority(self,
                      seq_group: SequenceGroup) -> Tuple[Optional[int], float]:
        """ Get the priority of the sequence group.
        Highest preference to user-defined priority, followed by arrival time.
        Args:
            seq_group: The sequence group input.
        Returns:
            The priority of the sequence group.
        """
        return seq_group.priority, seq_group.arrival_time

    def _schedule_priority_preemption(
        self,
        budget: SchedulingBudget,
    ) -> int:
        """Sorts waiting and running queue. Also, force preempt requests
        from the running queue if their priority is lower.
        Priority-based preemption is used with the priority policy.
        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
        Returns:
            A count of priority-based preemptions.
        """

        waiting_queue = self.waiting

        running_queue = deque(sorted(self.running, key=self._get_priority))

        blocks_to_swap_out: List[Tuple[int, int]] = []
        force_preemption_count = 0

        if waiting_queue:
            seq_group = waiting_queue.popleft()
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens_uncached, _ = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.WAITING, False, budget))

            #Only preempt if priority inversion exists
            while running_queue and self._get_priority(
                    running_queue[-1]) > self._get_priority(seq_group):
                #Only preempt if waiting sequence cannot be allocated
                can_allocate = self.block_manager.can_allocate(seq_group)
                if (num_new_tokens_uncached > 0
                        and can_allocate == AllocStatus.OK
                        and budget.can_schedule(
                            num_new_tokens=num_new_tokens_uncached,
                            num_new_seqs=num_new_seqs,
                        )):
                    break

                #Adjust budget to remove the victim sequence group
                vseq_group = running_queue.pop()
                num_running_tokens_uncached, _ = (
                    self._get_num_new_uncached_and_cached_tokens(
                        vseq_group, SequenceStatus.RUNNING, False, budget))
                budget.subtract_num_batched_tokens(
                    vseq_group.request_id, num_running_tokens_uncached)
                num_running_seqs = vseq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(vseq_group.request_id,
                                         num_running_seqs)

                #Preempt out the victim sequence group
                self._preempt(vseq_group, blocks_to_swap_out)
                waiting_queue.appendleft(vseq_group)
                force_preemption_count += 1
            #Put the sequence back into the waiting queue
            waiting_queue.appendleft(seq_group)

        waiting_queue = deque(sorted(waiting_queue, key=self._get_priority))

        self.waiting = waiting_queue
        self.running = running_queue
        #logger.info(f" iterative force_preemption_count={force_preemption_count}")
        #logger.info(f" waiting={len(self.waiting)}, running={len(self.running)}")
        return force_preemption_count

    def _schedule_prefills_separated(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerPrefillOutputs:
        """
        Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerPrefillOutputs.
        """
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []

        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        current_rel_id = None
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.WAITING, enable_chunking,
                    budget))
            num_new_tokens = num_new_tokens_uncached + num_new_tokens_cached

            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_lookahead_slots: int = 0
            if self.scheduler_config.is_multi_step and enable_chunking:
                num_lookahead_slots = self._get_num_lookahead_slots(
                    True, enable_chunking)

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(
                seq_group, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) + lookahead slots (%d) is "
                    "too long and exceeds the capacity of block_manager",
                    num_new_tokens, num_lookahead_slots)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            if (budget.num_batched_tokens >=
                    self.scheduler_config.max_num_batched_tokens):
                # We've reached the budget limit - since there might be
                # continuous prefills in the running queue, we should break
                # to avoid scheduling any new prefills.
                break

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                break

            # check rel_id and skip 
            if current_rel_id is None:
                current_rel_id = seq_group.rel_id
            else:
                if current_rel_id != seq_group.rel_id:
                    break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)

            if enable_chunking and self.scheduler_config.is_multi_step:
                blocks_to_copy: List[Tuple[int, int]] = []
                # init_multi_step_from_lookahead_slots happens in append_slots
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                # This assert will trip when a copy-on-write happens. This is
                # not a concern as the very first sequence-group block
                # allocation happens above. Still, we have the assert to
                # catch any edge-cases.
                assert not blocks_to_copy
            else:
                seq_group.init_multi_step_from_lookahead_slots(
                    num_lookahead_slots,
                    num_scheduler_steps=self.scheduler_config.
                    num_scheduler_steps,
                    is_multi_step=self.scheduler_config.is_multi_step,
                    enable_chunking=enable_chunking)

            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True, enable_chunking=enable_chunking))

    def _revert_prefill_seqs(
        self,
        seq_groups,
        budget: SchedulingBudget,
        record_budget_tx
    ):
        """
        deprecated
        remove the influence of every added seq_groups
        such revert is required since we need the next batch content to
        determine whether it is beneficial to overall latency

        related area
        1. self._allocate_and_set_running ==> deallocate and set waiting
        2. sg.init_multi_step_from_lookahead_slots ==> ignore since we assert no multi-step
        3. budget add_seqs and add_tokens ==> remove them
        """
        # first deallocate and set waiting
        for ssg in seq_groups:
            seq_group = ssg.seq_group
            """
            self.block_manager.allocate(seq_group)
            for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
                seq.status = SequenceStatus.RUNNING
            """
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq.status = SequenceStatus.WAITING
            for seq in seq_group.get_seqs():
                self.block_manager.free(seq)

        # then recover budget
        for info in record_budget_tx:
            req_id, nbt, nct, nns = info
            budget.subtract_num_batched_tokens(
                req_id,
                num_batched_tokens=nbt,
                num_cached_tokens=nct,
            )
            budget.subtract_num_seqs(req_id, nns)
            
    def _mimic_next_prefill_batch(self, budget, candidate_rel_id, NFT_tokens):
        """
        find the next prefill batch (a list of seq_groups)
        * do not exceed #tokens constraint
        * do not exceed #seqs constraint
        * do not exceed free #blocks in block_manager
        * only contain one rel_id

        since here at most one batch, we do not use approx like first/subsequent cache hit ratio
        """
        largest_candidate_set = [x for x in self.waiting if x.rel_id == candidate_rel_id]
        first_cache_miss_ratio, _ = self._compute_cache_miss(largest_candidate_set)

        # constraints
        GPU_free_num_tokens = self.block_manager.get_num_free_gpu_blocks() * self.block_manager.block_size
        cap_num_tokens = min(budget.remaining_token_budget(), GPU_free_num_tokens - NFT_tokens)
        cap_num_seqs = budget.remaining_nseq_budget()

        # mimic add 
        final_prefill_batch = []
        cur_batch_prefill_tokens = 0
        cur_batch_num_seqs = 0
        for sg in largest_candidate_set:
            if cur_batch_num_seqs + 1 > cap_num_seqs:
                break
            uncache_tokens, _ = self._pure_get_num_new_uncached_and_cached_tokens(
                    sg, SequenceStatus.WAITING)
            if cur_batch_prefill_tokens + uncache_tokens > cap_num_tokens:
                break
            cur_batch_num_seqs += 1
            cur_batch_prefill_tokens += uncache_tokens
            final_prefill_batch.append(sg)
         
        return final_prefill_batch, cur_batch_prefill_tokens


    def _rq_almost_done(self, target_rid):
        for sg in self.running:
            if sg.rel_id == target_rid:
                remaining_steps = sg.sampling_params.max_tokens - sg.get_seqs()[0].get_output_len()
                if remaining_steps < 10:
                    logger.info(f"running highest rQ {target_rid} almost done (remaining steps: {remaining_steps})")
                    return True
        return False

    def _schedule_prefills_ada_pabs(
        self,
        budget: SchedulingBudget,
        waiting_relid_cost,
        running_relid_cost,
        overlap_policy,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerPrefillOutputs:
        """Schedule sequence groups that are in prefill stage.
        waiting_relid_cost: all relQuery in the waiting queue and its corresponding priority
        running_relid_cost: all relQueries that only exist in running queue and its priority
        both are List[(relid, priority)]

        overlap_policy: 
        * ovlp: vllm style, overlap P & D as much as possible regardless of inter/intra-rQ
        * islt: isolated inter-rQ PD (in priority, it means the prefill of lower-prio rQ are deferred)
        * ada: our proposed adaptive PD control for inter-relQuery

        ada logic:
        * before prefill while loop, first detect inter/intra moment
        * if intra/cold start/preempt: ovlp + only_one_rel_id constraint
        * if inter:
                1. construct a plausible next prefill batch
                2. predict the benefit of do_overlap or not
            if do_overlap = False: islt + only_one_rel_id constraint
            if do_overlap = True: ovlp + only_one_rel_id constraint

        """
        assert not self.lora_enabled
        # TODO: if chunking or multistep, need to properly handle init_multi_step_from_lookahead_slots
        assert not enable_chunking and not self.scheduler_config.is_multi_step 
        assert overlap_policy in ["ada", "ovlp", "islt"]
        logger.info(f"overlap policy: {overlap_policy}")

        # obtain number of blocks to reserve for necessary future tokens 
        NFT = 0
        for sg in self.running:
            max_output_tokens = sg.sampling_params.max_tokens
            cur_seq = sg.get_seqs(status=SequenceStatus.RUNNING)[0]
            cur_output_tokens = cur_seq.get_output_len()
            NFT += (max_output_tokens - cur_output_tokens)
        NFT_blocks = NFT // self.block_manager.block_size + 1

        ########################### new logic start
        only_one_rel_id = False
        # ada policy will fall into either ovlp or islt before prefill loop
        tmp_overlap_policy = overlap_policy if overlap_policy in ["ovlp", "islt"] else None
        if overlap_policy in ["ada", "islt"]:
            # both need to detect inter-rQ or not
            only_one_rel_id = True
            contained_rel_id = None
            is_inter = False
            if len(waiting_relid_cost) and len(running_relid_cost) and running_relid_cost[0][1] < waiting_relid_cost[0][1]:
                """
                if len(waiting_relid_cost) == 0: nothing to decide, must be decode
                if len(running_relid_cost) == 0: either none is running, or running rQ has unfinished reqs in waiting, intra-rQ
                if running_relid_cost[0][1] > waiting_relid_cost[0][1]: preempt a decoding rQ
                actually this criteria tends to overlap: waiting_relid_cost[0]'s priority does not include its running part
                """
                is_inter = True
                logger.info(f"inter-rQ")
            else:
                logger.info(f"intra-rQ / cold start / preempt")

            # ada need to determine further action
            if is_inter and overlap_policy == "ada":
                """
                (1) if running rQ with higher prio has few decoding steps, islt
                (2) otherwise, construct the next prefill batch, and check delta latency
                """
                tic = time.perf_counter()
                if self._rq_almost_done(running_relid_cost[0][0]):
                    # islt to decode since running highest prio rQ almost done
                    delta_latency = 111
                else:
                    mimic_prefill_batch, mimic_prefill_tokens = self._mimic_next_prefill_batch(budget,
                            waiting_relid_cost[0][0], NFT)
                    if len(mimic_prefill_batch) == 0:
                        # islt to decode since no KV cache space
                        delta_latency = 222
                    else:
                        # check delta latency
                        delta_latency = self.predict_delta_latency(mimic_prefill_batch, waiting_relid_cost, running_relid_cost, mimic_prefill_tokens)
                do_overlap = delta_latency < 0
                if do_overlap:
                    tmp_overlap_policy = "ovlp"
                else:
                    tmp_overlap_policy = "islt"
                dur = time.perf_counter() - tic
                logger.info(f"delta_latency: {delta_latency}, do_overlap: {do_overlap}, overhead: {dur:.5f}")
        ########################### new logic end

        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []
        waiting_queue = self.waiting
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]
            ########################### new logic start
            # one batch can only contain one rel_id
            if only_one_rel_id:
                if contained_rel_id is None:
                    contained_rel_id = seq_group.rel_id
                else:
                    if seq_group.rel_id != contained_rel_id:
                        logger.info(f"one rel_id constraint break prefill loop: attempted {seq_group.rel_id} contained {contained_rel_id}")
                        break
            # islt logic: stop prefill for inter-rQ
            if tmp_overlap_policy == "islt":
                if is_inter:
                    break
            ########################### new logic end

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.WAITING, enable_chunking,
                    budget))
            num_new_tokens = num_new_tokens_uncached + num_new_tokens_cached

            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_lookahead_slots: int = 0
            if self.scheduler_config.is_multi_step and enable_chunking:
                num_lookahead_slots = self._get_num_lookahead_slots(
                    True, enable_chunking)

            ########################### new logic start
            # before checking can_allocate, check reserved space for NFT
            cur_free_num_blocks = self.block_manager.get_num_free_gpu_blocks()
            if cur_free_num_blocks < NFT_blocks:
                logger.info(f"NFT break prefill loop: NFT_blocks: {NFT_blocks}, current free blocks: {cur_free_num_blocks}")
                break
            ########################### new logic end

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(
                seq_group, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) + lookahead slots (%d) is "
                    "too long and exceeds the capacity of block_manager",
                    num_new_tokens, num_lookahead_slots)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            if (budget.num_batched_tokens >=
                    self.scheduler_config.max_num_batched_tokens):
                # We've reached the budget limit - since there might be
                # continuous prefills in the running queue, we should break
                # to avoid scheduling any new prefills.
                break

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                break

            # Can schedule this request.
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)

            if enable_chunking and self.scheduler_config.is_multi_step:
                blocks_to_copy: List[Tuple[int, int]] = []
                # init_multi_step_from_lookahead_slots happens in append_slots
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                # This assert will trip when a copy-on-write happens. This is
                # not a concern as the very first sequence-group block
                # allocation happens above. Still, we have the assert to
                # catch any edge-cases.
                assert not blocks_to_copy
            else:
                seq_group.init_multi_step_from_lookahead_slots(
                    num_lookahead_slots,
                    num_scheduler_steps=self.scheduler_config.num_scheduler_steps,
                    is_multi_step=self.scheduler_config.is_multi_step,
                    enable_chunking=enable_chunking)

            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True, enable_chunking=enable_chunking))

    def predict_delta_latency(self, next_prefill_batch, waiting_relid_cost, running_relid_cost, prefill_num_tokens):
        """
        if insert the next_prefill_batch of rQ2 before the decoding of rQ1, give the delta of latency
        * for running high prio rQ1: latency increased by (1) prefill time (2) delta decode time
        * for waiting low prio rQ2: latency reduced by (1) less decode iteration - delta decode time
        * for other waiting rQ: rQ2's finish time is moved ahead by x, then Nx is reduced time

        due to iterative decision, there maybe multiple rQ1
        """
        slope_decode, intercept_decode = self.scheduler_config.info_decode
        slope_prefill, intercept_prefill = self.scheduler_config.info_prefill
        # next prefill batch's time
        prefill_batch_time = prefill_num_tokens * slope_prefill + intercept_prefill
        logger.info(f"next prefill batch #tokens: {prefill_num_tokens}, #seqs: {len(next_prefill_batch)}, estimate time: {prefill_batch_time:.5f} s")
        # delta decode time per step
        delta_decode_per_step = len(next_prefill_batch) * slope_decode
        # new added batch's decode len, subtract 1 for prefill
        new_added_output_len = next_prefill_batch[0].sampling_params.max_tokens - 1
        logger.info(f"delta decode per step: {delta_decode_per_step:.5f} s, new_added_output_len: {new_added_output_len}")

        running_queries = defaultdict(list)
        for sg in self.running:
            running_queries[sg.rel_id].append(sg)
        waiting_queries = defaultdict(list)
        for sg in self.waiting:
            waiting_queries[sg.rel_id].append(sg)

        # increased latency of rQ1
        increased_rQ1_latency = 0
        record_rQ1_max_output_len = 0
        for rid, _ in running_relid_cost:
            # prefill incurred latency
            increased_rQ1_latency += prefill_batch_time
            # decode incurred latency
            sgs = running_queries[rid]
            max_output_len = max(sg.sampling_params.max_tokens - sg.get_seqs()[0].get_output_len() for sg in sgs)
            record_rQ1_max_output_len = max(record_rQ1_max_output_len, max_output_len)
            # if new is longer, then full cost, otherwise only partial cost
            increased_rQ1_latency += delta_decode_per_step * min(new_added_output_len, max_output_len)
            logger.info(f"rQ {rid}, max output len {max_output_len}, increased latency: {increased_rQ1_latency:.5f}")

        # decreased latency of rQ2
        # the insert of prefill batch save many steps of decoding! but the delta decode overhead must be counted
        # rQ2_orig_decode_per_step_time = slope_decode * len(next_prefill_batch) + intercept_decode
        # decreased_rQ2_latency += (rQ2_orig_decode_per_step_time - delta_decode_per_step) * actual saving steps
        decreased_rQ2_latency = min(new_added_output_len, record_rQ1_max_output_len) * intercept_decode

        # waiting ones
        decreased_waiting_latency = (len(waiting_relid_cost) - 1) * decreased_rQ2_latency

        logger.info(f"rQ2 & other waitings decreased latency: {decreased_rQ2_latency:.5f} & {decreased_waiting_latency:.5f}")

        # if return < 0, means insert this prefill brings less latency
        return increased_rQ1_latency - decreased_rQ2_latency - decreased_waiting_latency

    def predict_with_overlap_latency(self, next_prefill_batch, waiting_relid_cost, prefill_num_tokens):
        offset = 0
        total_latency = 0

        slope_decode, intercept_decode = self.scheduler_config.info_decode
        slope_prefill, intercept_prefill = self.scheduler_config.info_prefill

        # 1. next prefill batch's time
        offset += prefill_num_tokens * slope_prefill + intercept_prefill
        logger.info(f"next prefill batch #tokens: {prefill_num_tokens}, #seqs: {len(next_prefill_batch)}, estimate time: {offset:.5f} s")

        # 2. running relqueries' time, offseted by prefill's time
        WIDs = set(x[0] for x in waiting_relid_cost)
        #    filter out preempted ones
        running_queries = defaultdict(list)
        for sg in self.running:
            if sg.rel_id in WIDs:
                continue
            running_queries[sg.rel_id].append(sg)
        #    obtain leftover length for each rQ
        running_queries_leftover_length = dict()
        for rid, sgs in running_queries.items():
            max_output_len = max(sg.sampling_params.max_tokens - sg.get_seqs()[0].get_output_len()
                    for sg in sgs)
            running_queries_leftover_length[rid] = max_output_len
        #    calc latency for each rQ
        decoding_bs = len(self.running) + len(next_prefill_batch)
        logger.info(f"anticipated decode bs: {decoding_bs}, single step time: {slope_decode * decoding_bs + intercept_decode:.5f} s")
        max_lat = -1
        for rid, max_output_len in running_queries_leftover_length.items():
            cur_latency = offset + max_output_len * (slope_decode * decoding_bs + intercept_decode)
            logger.info(f"-- rid {rid} latency: {cur_latency}")
            total_latency += cur_latency
            max_lat = max(max_lat, cur_latency)
        offset = max_lat

        # 3. calculate latency for each waiting query
        """
        now waiting is changed slightly: the most fronted relQuery has fewer (down to 0) reqs
        due to next prefill batch; for other rQs, nothing is changed
        """
        remaining_front_rq = [sg for sg in self.waiting if sg.rel_id == next_prefill_batch[0].rel_id]
        remaining_front_rq = remaining_front_rq[len(next_prefill_batch):]
        # leftover partial reqs' workload (decoding)
        max_output_len_running = max(running_queries_leftover_length.values())
        inserted_batch_output_len = next_prefill_batch[0].sampling_params.max_tokens - 1
        partial_workload_front_rq = 0
        if max_output_len_running < inserted_batch_output_len:
            # need extra decoding steps for next_prefill_batch
            partial_workload_front_rq += (inserted_batch_output_len - max_output_len_running) * \
                    (slope_decode * len(next_prefill_batch) + intercept_decode)
        # leftover intact reqs' workload (prefill + decode)
        # attention: here we should use subsequent cache hit ratio: one prefill is executed already
        intact_workload_front_rq = self._calc_single_query_priority(remaining_front_rq, subsequent_cache=True)
        logger.info(f"front-waiting-rQ workload: partial {partial_workload_front_rq} intact {intact_workload_front_rq}")
        whole_workload_front_rq = partial_workload_front_rq + intact_workload_front_rq

        # include each rid's time and check
        # for the most front rQ, its workload is changed
        new_tp = (waiting_relid_cost[0][0], whole_workload_front_rq)
        old_tp = (waiting_relid_cost[0][0], waiting_relid_cost[0][1])
        waiting_relid_cost[0] = new_tp

        # for each of waiting rQ, its latency is: waiting time (offset) + execution time
        for rid, duration in waiting_relid_cost:
            cur_latency = offset + duration
            logger.info(f"-- rid {rid} latency: {cur_latency}")
            total_latency += cur_latency
            offset += duration

        # restore waiting_relid_cost
        waiting_relid_cost[0] = old_tp

        return total_latency

    def predict_no_overlap_latency(self, waiting_relid_cost):
        highest_prio_in_waiting = waiting_relid_cost[0][1]
        WIDs = set(x[0] for x in waiting_relid_cost)
        running_queries = defaultdict(list)
        for sg in self.running:
            if sg.rel_id in WIDs:
                # ignore unfinished waiting ones
                continue
            running_queries[sg.rel_id].append(sg)

        running_queries_leftover_length = dict()
        for rid, sgs in running_queries.items():
            max_output_len = max(sg.sampling_params.max_tokens - sg.get_seqs()[0].get_output_len()
                    for sg in sgs)
            running_queries_leftover_length[rid] = max_output_len

        slope_decode, intercept_decode = self.scheduler_config.info_decode
        total_latency = 0
        offset = 0

        # first sum the latency of running ones, here no waiting
        decoding_bs = len(self.running)
        for rid, max_output_len in running_queries_leftover_length.items():
            cur_latency = max_output_len * (slope_decode * decoding_bs + intercept_decode)
            total_latency += cur_latency
            offset = max(offset, cur_latency)

        # for each of waiting rQ, its latency is: waiting time (offset) + execution time
        for rid, duration in waiting_relid_cost:
            cur_latency = offset + duration
            total_latency += cur_latency
            offset += duration
        return total_latency

    def _schedule_prefills_iso_pabs(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerPrefillOutputs:
        """Schedule sequence groups that are in prefill stage.
        """
        # by default we insert prefill: (1) intra-relQuery (2) empty waiting (3) empty running
        insert_prefill = True
        if len(self.waiting):
            # highest priority rel_id and cost in waiting queue
            WID = self.waiting[0].rel_id # self.waiting already sorted before schedule_running/_prefill
            RIDs = [x.rel_id for x in self.running]
            if len(RIDs) and WID not in RIDs:
                # inter-relQuery execution
                insert_prefill = False

        # obtain number of blocks to reserve for necessary future tokens 
        NFT = 0
        for sg in self.running:
            max_output_tokens = sg.sampling_params.max_tokens
            cur_seq = sg.get_seqs(status=SequenceStatus.RUNNING)[0]
            cur_output_tokens = cur_seq.get_output_len()
            NFT += (max_output_tokens - cur_output_tokens)
        NFT_blocks = NFT // self.block_manager.block_size + 1

        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []

        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            # for inter-relQuery case, delay prefill
            if not insert_prefill:
                break

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.WAITING, enable_chunking,
                    budget))
            num_new_tokens = num_new_tokens_uncached + num_new_tokens_cached

            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_lookahead_slots: int = 0
            if self.scheduler_config.is_multi_step and enable_chunking:
                num_lookahead_slots = self._get_num_lookahead_slots(
                    True, enable_chunking)

            # before checking can_allocate, check reserved space for NFT
            if self.block_manager.get_num_free_gpu_blocks() < NFT_blocks:
                break

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(
                seq_group, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) + lookahead slots (%d) is "
                    "too long and exceeds the capacity of block_manager",
                    num_new_tokens, num_lookahead_slots)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            if (budget.num_batched_tokens >=
                    self.scheduler_config.max_num_batched_tokens):
                # We've reached the budget limit - since there might be
                # continuous prefills in the running queue, we should break
                # to avoid scheduling any new prefills.
                break

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)

            if enable_chunking and self.scheduler_config.is_multi_step:
                blocks_to_copy: List[Tuple[int, int]] = []
                # init_multi_step_from_lookahead_slots happens in append_slots
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                # This assert will trip when a copy-on-write happens. This is
                # not a concern as the very first sequence-group block
                # allocation happens above. Still, we have the assert to
                # catch any edge-cases.
                assert not blocks_to_copy
            else:
                seq_group.init_multi_step_from_lookahead_slots(
                    num_lookahead_slots,
                    num_scheduler_steps=self.scheduler_config.
                    num_scheduler_steps,
                    is_multi_step=self.scheduler_config.is_multi_step,
                    enable_chunking=enable_chunking)

            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True, enable_chunking=enable_chunking))

    def _schedule_prefills_ipabs(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerPrefillOutputs:
        """Schedule sequence groups that are in prefill stage.
        """

        # obtain current relQuery id
        running_rel_id = None
        running_output_len = 0
        if self.running:
            running_rel_id = self.running[0].rel_id
            running_output_len = self.running[0].sampling_params.max_tokens# - self.running[0].get_seqs()[0].get_output_len()

        # obtain number of blocks to reserve for necessary future tokens 
        NFT = 0
        for sg in self.running:
            max_output_tokens = sg.sampling_params.max_tokens
            cur_seq = sg.get_seqs(status=SequenceStatus.RUNNING)[0]
            cur_output_tokens = cur_seq.get_output_len()
            NFT += (max_output_tokens - cur_output_tokens)
        NFT_blocks = NFT // self.block_manager.block_size + 1

        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []

        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            # before scheduling, check whether have other relQuery is decoding & decoding length is short
            if running_rel_id is not None and seq_group.rel_id != running_rel_id and running_output_len <= 10:
                logger.info(f"prevent prefill, running_rel_id: {running_rel_id}, running_output_len: {running_output_len}")
                break

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.WAITING, enable_chunking,
                    budget))
            num_new_tokens = num_new_tokens_uncached + num_new_tokens_cached

            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_lookahead_slots: int = 0
            if self.scheduler_config.is_multi_step and enable_chunking:
                num_lookahead_slots = self._get_num_lookahead_slots(
                    True, enable_chunking)

            # before checking can_allocate, check reserved space for NFT
            if self.block_manager.get_num_free_gpu_blocks() < NFT_blocks:
                break

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(
                seq_group, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) + lookahead slots (%d) is "
                    "too long and exceeds the capacity of block_manager",
                    num_new_tokens, num_lookahead_slots)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            if (budget.num_batched_tokens >=
                    self.scheduler_config.max_num_batched_tokens):
                # We've reached the budget limit - since there might be
                # continuous prefills in the running queue, we should break
                # to avoid scheduling any new prefills.
                break

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)

            if enable_chunking and self.scheduler_config.is_multi_step:
                blocks_to_copy: List[Tuple[int, int]] = []
                # init_multi_step_from_lookahead_slots happens in append_slots
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                # This assert will trip when a copy-on-write happens. This is
                # not a concern as the very first sequence-group block
                # allocation happens above. Still, we have the assert to
                # catch any edge-cases.
                assert not blocks_to_copy
            else:
                seq_group.init_multi_step_from_lookahead_slots(
                    num_lookahead_slots,
                    num_scheduler_steps=self.scheduler_config.
                    num_scheduler_steps,
                    is_multi_step=self.scheduler_config.is_multi_step,
                    enable_chunking=enable_chunking)

            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True, enable_chunking=enable_chunking))


    def _schedule_prefills_pabs(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerPrefillOutputs:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerPrefillOutputs.
        """

        # obtain number of blocks to reserve for necessary future tokens 
        NFT = 0
        for sg in self.running:
            max_output_tokens = sg.sampling_params.max_tokens
            cur_seq = sg.get_seqs(status=SequenceStatus.RUNNING)[0]
            cur_output_tokens = cur_seq.get_output_len()
            NFT += (max_output_tokens - cur_output_tokens)
        NFT_blocks = NFT // self.block_manager.block_size + 1

        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []

        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.WAITING, enable_chunking,
                    budget))
            num_new_tokens = num_new_tokens_uncached + num_new_tokens_cached

            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_lookahead_slots: int = 0
            if self.scheduler_config.is_multi_step and enable_chunking:
                num_lookahead_slots = self._get_num_lookahead_slots(
                    True, enable_chunking)

            # before checking can_allocate, check reserved space for NFT
            if self.block_manager.get_num_free_gpu_blocks() < NFT_blocks:
                break

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(
                seq_group, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) + lookahead slots (%d) is "
                    "too long and exceeds the capacity of block_manager",
                    num_new_tokens, num_lookahead_slots)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            if (budget.num_batched_tokens >=
                    self.scheduler_config.max_num_batched_tokens):
                # We've reached the budget limit - since there might be
                # continuous prefills in the running queue, we should break
                # to avoid scheduling any new prefills.
                break

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)

            if enable_chunking and self.scheduler_config.is_multi_step:
                blocks_to_copy: List[Tuple[int, int]] = []
                # init_multi_step_from_lookahead_slots happens in append_slots
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                # This assert will trip when a copy-on-write happens. This is
                # not a concern as the very first sequence-group block
                # allocation happens above. Still, we have the assert to
                # catch any edge-cases.
                assert not blocks_to_copy
            else:
                seq_group.init_multi_step_from_lookahead_slots(
                    num_lookahead_slots,
                    num_scheduler_steps=self.scheduler_config.
                    num_scheduler_steps,
                    is_multi_step=self.scheduler_config.is_multi_step,
                    enable_chunking=enable_chunking)

            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True, enable_chunking=enable_chunking))


    def _schedule_prefills(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerPrefillOutputs:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerPrefillOutputs.
        """
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []

        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.WAITING, enable_chunking,
                    budget))
            num_new_tokens = num_new_tokens_uncached + num_new_tokens_cached

            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_lookahead_slots: int = 0
            if self.scheduler_config.is_multi_step and enable_chunking:
                num_lookahead_slots = self._get_num_lookahead_slots(
                    True, enable_chunking)

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(
                seq_group, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) + lookahead slots (%d) is "
                    "too long and exceeds the capacity of block_manager",
                    num_new_tokens, num_lookahead_slots)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            if (budget.num_batched_tokens >=
                    self.scheduler_config.max_num_batched_tokens):
                # We've reached the budget limit - since there might be
                # continuous prefills in the running queue, we should break
                # to avoid scheduling any new prefills.
                break

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)

            if enable_chunking and self.scheduler_config.is_multi_step:
                blocks_to_copy: List[Tuple[int, int]] = []
                # init_multi_step_from_lookahead_slots happens in append_slots
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                # This assert will trip when a copy-on-write happens. This is
                # not a concern as the very first sequence-group block
                # allocation happens above. Still, we have the assert to
                # catch any edge-cases.
                assert not blocks_to_copy
            else:
                seq_group.init_multi_step_from_lookahead_slots(
                    num_lookahead_slots,
                    num_scheduler_steps=self.scheduler_config.
                    num_scheduler_steps,
                    is_multi_step=self.scheduler_config.is_multi_step,
                    enable_chunking=enable_chunking)

            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True, enable_chunking=enable_chunking))

    def _combine_prefills(self, all_prefills):
        out_prefills = SchedulerPrefillOutputs.create_empty()
        if len(all_prefills) != 0:
            for p in all_prefills:
                out_prefills.seq_groups += p.seq_groups
                out_prefills.ignored_seq_groups += p.ignored_seq_groups
                assert p.num_lookahead_slots == 0
        if len(out_prefills.seq_groups) > 0:
            self.prev_prompt = True
        return out_prefills

    def _update_query_priority_tc(self):
        ts = time.perf_counter()
        waiting_queries = defaultdict(list)
        for sg in self.waiting:
            waiting_queries[sg.rel_id].append(sg)

        record_relid_cost = []
        for rel_id, sg_list in waiting_queries.items():
            cost = 0
            for sg in sg_list:
                # output tokens
                cost += sg.sampling_params.max_tokens
                # input tokens
                cost += sg.get_seqs()[0].get_prompt_len()
            for sg in sg_list:
                # update priority
                sg.priority = cost
            record_relid_cost.append((rel_id, cost))

        new_waiting = []
        for rel_id, _ in sorted(record_relid_cost, key=lambda x:x[1]):
            new_waiting += waiting_queries[rel_id]
        self.waiting = deque(new_waiting)
        
        elapsed_time = time.perf_counter() - ts
        logger.info(f"priority updating overhead in seconds: {elapsed_time}")


    def _compute_cache_miss(self, sg_list, enforce_second=False):
        if len(sg_list) == 0:
            raise ValueError

        sample_size = min(10, len(sg_list))

        first_batch_uncached_tokens = 0
        first_batch_cached_tokens = 0
        for sg in sg_list[:sample_size]:
            num_uncached_new_tokens, num_cached_new_tokens = \
                self._pure_get_num_new_uncached_and_cached_tokens(
                    sg, SequenceStatus.WAITING)
            first_batch_uncached_tokens += num_uncached_new_tokens
            first_batch_cached_tokens += num_cached_new_tokens

        first_batch_cache_miss_ratio = first_batch_uncached_tokens / (
                first_batch_uncached_tokens + first_batch_cached_tokens)

        return (first_batch_cache_miss_ratio, first_batch_cache_miss_ratio)
        estimated_total_prefill_tokens = first_batch_uncached_tokens * len(sg) / sample_size if sample_size > 10 else first_batch_uncached_tokens
        if not enforce_second and estimated_total_prefill_tokens <= self.scheduler_config.max_num_batched_tokens:
            # one batch is sufficient to prefill all
            return (first_batch_cache_miss_ratio, first_batch_cache_miss_ratio)

        if sample_size > len(sg_list) / 2:
            # to avoid next_batch has no content...
            sample_size = int(sample_size / 2)
            if sample_size == 0:
                # too few samples, directly return 
                return (first_batch_cache_miss_ratio, first_batch_cache_miss_ratio)
            #assert sample_size

        # calculate subsequent_batch_cache_miss_ratio
        sampled_first_batch_token_ids_set = self._obtain_token_ids_set_from_sg_list(sg_list[:sample_size])
        sampled_next_batch_token_ids_set = self._obtain_token_ids_set_from_sg_list(sg_list[sample_size:sample_size*2])
        overlap_token_ids = sampled_next_batch_token_ids_set.intersection(sampled_first_batch_token_ids_set)
        overlap_ratio = len(overlap_token_ids) / len(sampled_next_batch_token_ids_set)
        subsequent_batch_cache_miss_ratio = max(overlap_ratio, first_batch_cache_miss_ratio)
        return (first_batch_cache_miss_ratio, subsequent_batch_cache_miss_ratio)

    def _calc_single_query_priority_cd(self, target_sg_list, enforce_second=False, subsequent_cache=False):
        if len(target_sg_list) == 0:
            return 0
        assert len(set(sg.rel_id for sg in target_sg_list)) == 1, "Not single relQuery"
        assert not enforce_second
        assert not subsequent_cache

        slope_prefill, intercept_prefill = self.scheduler_config.info_prefill
        slope_decode, intercept_decode = self.scheduler_config.info_decode

        first_cache_miss_ratio, _ = self._compute_cache_miss(target_sg_list, enforce_second)

        cap_gpu_kv_tokens = self.block_manager.num_total_gpu_blocks * self.block_manager.block_size

        """
        multiple prefill batch forms one decode batch
        """
        cost = 0
        cur_prefill_batch_tokens = 0
        accum_prefill_tokens = 0
        cur_decode_batch_seqs = 0
        output_len = target_sg_list[0].sampling_params.max_tokens - 1
        for sg in target_sg_list:
            uncache_tokens = sg.get_seqs()[0].get_prompt_len() * first_cache_miss_ratio

            if accum_prefill_tokens + uncache_tokens > cap_gpu_kv_tokens:
                """
                GPU space is full, need to flush:
                    * execute current prefill batch
                    * execute decode batches for previously prefilled requests 
                """
                cost += cur_prefill_batch_tokens * slope_prefill + intercept_prefill
                cost += (cur_decode_batch_seqs * slope_decode + intercept_decode) * output_len
                # reset 
                cur_prefill_batch_tokens = 0
                cur_decode_batch_seqs = 0
                accum_prefill_tokens = 0

            # insert request into GPU space
            if cur_prefill_batch_tokens + uncache_tokens > self.scheduler_config.max_num_batched_tokens:
                # prefill batch full, execute one prefill batch
                cost += cur_prefill_batch_tokens * slope_prefill + intercept_prefill
                # reset
                cur_prefill_batch_tokens = 0

            cur_prefill_batch_tokens += uncache_tokens
            accum_prefill_tokens += uncache_tokens
            cur_decode_batch_seqs += 1

        # leftover
        if cur_decode_batch_seqs:
            cost += cur_prefill_batch_tokens * slope_prefill + intercept_prefill
            cost += (cur_decode_batch_seqs * slope_decode + intercept_decode) * output_len

        return cost


    def _calc_single_query_priority(self, target_sg_list, enforce_second=False, subsequent_cache=False):
        """
        given a list, return the abs priority
        """
        if len(target_sg_list) == 0:
            return 0
        assert len(set(sg.rel_id for sg in target_sg_list)) == 1, "Not single relQuery"

        slope_prefill, intercept_prefill = self.scheduler_config.info_prefill
        slope_decode, intercept_decode = self.scheduler_config.info_decode

        first_cache_miss_ratio, subsequent_cache_miss_ratio = self._compute_cache_miss(target_sg_list, enforce_second)
        #logger.info(f"rid: {target_sg_list[0].rel_id}, first/subsequent cache miss_ratio: {first_cache_miss_ratio:.2f} / {subsequent_cache_miss_ratio:.2f}")

        if subsequent_cache:
            first_cache_miss_ratio = subsequent_cache_miss_ratio

        cost = 0
        cur_batch_prefill_tokens = 0
        cur_batch_num_seqs = 0
        cur_batch_max_output_len = 0
        for sg in target_sg_list:
            if cost == 0:
                # first batch
                uncache_tokens = sg.get_seqs()[0].get_prompt_len() * first_cache_miss_ratio
            else:
                # subsequent batch
                uncache_tokens = sg.get_seqs()[0].get_prompt_len() * subsequent_cache_miss_ratio

            #if cur_batch_prefill_tokens + uncache_tokens > self.scheduler_config.max_model_len:
            if cur_batch_prefill_tokens + uncache_tokens > self.scheduler_config.max_num_batched_tokens:
                # prefill cost
                cost += cur_batch_prefill_tokens * slope_prefill + intercept_prefill
                # decode cost
                cost += (cur_batch_num_seqs * slope_decode + intercept_decode) * cur_batch_max_output_len
                # reset
                cur_batch_prefill_tokens = 0
                cur_batch_num_seqs = 0
                cur_batch_max_output_len = 0
            # batching
            cur_batch_prefill_tokens += uncache_tokens
            cur_batch_num_seqs += 1
            #cur_batch_max_output_len = max(cur_batch_max_output_len, sg.sampling_params.max_tokens)
            cur_batch_max_output_len = max(cur_batch_max_output_len, sg.sampling_params.max_tokens - 1)

        # leftover
        if cur_batch_num_seqs:
            cost += cur_batch_prefill_tokens * slope_prefill + intercept_prefill
            cost += (cur_batch_num_seqs * slope_decode + intercept_decode) * cur_batch_max_output_len

        return cost


    def _update_waiting_priority_abs(self, staleness_bound=-1):
        slope_prefill, intercept_prefill = self.scheduler_config.info_prefill
        slope_decode, intercept_decode = self.scheduler_config.info_decode

        waiting_queries = defaultdict(list)
        for sg in self.waiting:
            waiting_queries[sg.rel_id].append(sg)

        record_relid_cost = []
        for rel_id, sg_list in waiting_queries.items():
            
            if rel_id in self.historical_cache_miss and self.historical_staleness[rel_id] <= staleness_bound:
                # reuse historical cache miss ratio
                first_cache_miss_ratio, subsequent_cache_miss_ratio = self.historical_cache_miss[rel_id]
                self.historical_staleness[rel_id] += 1
            else:
                # either first time, or exceed staleness bound, recompute cache miss 
                first_cache_miss_ratio, subsequent_cache_miss_ratio = self._compute_cache_miss(sg_list)
                self.historical_cache_miss[rel_id] = (first_cache_miss_ratio, subsequent_cache_miss_ratio)
                self.historical_staleness[rel_id] = 0

            cost = 0
            cur_batch_prefill_tokens = 0
            cur_batch_num_seqs = 0
            cur_batch_max_output_len = 0
            for sg in sg_list:
                if cost == 0:
                    # first batch
                    uncache_tokens = sg.get_seqs()[0].get_prompt_len() * first_cache_miss_ratio
                else:
                    # subsequent batch
                    uncache_tokens = sg.get_seqs()[0].get_prompt_len() * subsequent_cache_miss_ratio

                #if cur_batch_prefill_tokens + uncache_tokens > self.scheduler_config.max_model_len:
                if cur_batch_prefill_tokens + uncache_tokens > self.scheduler_config.max_num_batched_tokens:
                    # prefill cost
                    cost += cur_batch_prefill_tokens * slope_prefill + intercept_prefill
                    # decode cost
                    cost += (cur_batch_num_seqs * slope_decode + intercept_decode) * cur_batch_max_output_len
                    # reset 
                    cur_batch_prefill_tokens = 0
                    cur_batch_num_seqs = 0
                    cur_batch_max_output_len = 0
                # batching
                cur_batch_prefill_tokens += uncache_tokens
                cur_batch_num_seqs += 1
                #cur_batch_max_output_len = max(cur_batch_max_output_len, sg.sampling_params.max_tokens)
                cur_batch_max_output_len = max(cur_batch_max_output_len, sg.sampling_params.max_tokens-1)

            # leftover
            if cur_batch_num_seqs:
                cost += cur_batch_prefill_tokens * slope_prefill + intercept_prefill
                cost += (cur_batch_num_seqs * slope_decode + intercept_decode ) * cur_batch_max_output_len

            for sg in sg_list:
                # update priority
                sg.priority = cost
            record_relid_cost.append((rel_id, cost))

        sorted_record = sorted(record_relid_cost, key=lambda x:x[1])
        new_waiting = []
        for rel_id, _ in sorted_record:
            new_waiting += waiting_queries[rel_id]
        self.waiting = deque(new_waiting)

        return sorted_record

    def _update_running_priority_abs(self, waiting_relid_cost):
        slope_decode, intercept_decode = self.scheduler_config.info_decode
        wait_rq_dict = {rid:prio for rid, prio in waiting_relid_cost}
        running_queries = defaultdict(list)
        running_rq_cost = []
        for sg in self.running:
            running_queries[sg.rel_id].append(sg)
        for rid, sgs in running_queries.items():
            if rid in wait_rq_dict:
                # use waiting priority
                for sg in sgs:
                    sg.priority = wait_rq_dict[rid]
            else:
                # calculate new priority and update
                leftover_num_steps = max(sg.sampling_params.max_tokens - sg.get_seqs()[0].get_output_len()
                        for sg in sgs)
                """
                if we use len(self.running) as decoding bs, the priority scale is not consistent across
                waiting and running: in waiting, priority are calculated with assumption of isolated 
                execution, so here we use len(sgs) as decoding bs
                """
                #decoding_bs = len(self.running) 
                decoding_bs = len(sgs)
                leftover_cost = leftover_num_steps * (slope_decode * decoding_bs + intercept_decode)
                for sg in sgs:
                    sg.priority = leftover_cost
                running_rq_cost.append((rid, leftover_cost))
        sorted_record = sorted(running_rq_cost, key=lambda x:x[1])
        return sorted_record

    def _update_query_priority_bs(self, max_sim_steps=0):

        ts = time.perf_counter()
        slope_prefill, intercept_prefill = self.scheduler_config.info_prefill
        slope_decode, intercept_decode = self.scheduler_config.info_decode

        waiting_queries = defaultdict(list)
        for sg in self.waiting:
            waiting_queries[sg.rel_id].append(sg)

        record_relid_cost = []
        for rel_id, sg_list in waiting_queries.items():
            total_cost = 0

            # obtain first batch's cost and statistics
            first_batch_uncached_tokens = 0
            first_batch_cached_tokens = 0
            first_batch_num_seqs = 0
            first_batch_max_output_len = 0
            for sg in sg_list:
                num_uncached_new_tokens, num_cached_new_tokens = \
                    self._pure_get_num_new_uncached_and_cached_tokens(
                        sg, SequenceStatus.WAITING)
                if first_batch_uncached_tokens + num_uncached_new_tokens < self.scheduler_config.max_model_len:
                    # okay to add
                    first_batch_uncached_tokens += num_uncached_new_tokens
                    first_batch_cached_tokens += num_cached_new_tokens
                    first_batch_num_seqs += 1
                    first_batch_max_output_len = max(first_batch_max_output_len, sg.sampling_params.max_tokens)
                else:
                    break

            # first batch content determined, calculate cost
            total_cost += first_batch_uncached_tokens * slope_prefill + intercept_prefill
            total_cost += first_batch_max_output_len * (first_batch_num_seqs * slope_decode
                    + intercept_decode)

            # check for remaining requests
            if len(sg_list) > first_batch_num_seqs:
                first_batch_cache_miss_ratio = first_batch_uncached_tokens / (
                        first_batch_uncached_tokens + first_batch_cached_tokens)
                
                # estimate subsequent cache miss ratio
                if max_sim_steps == 0:
                    subsequent_cache_miss_ratio = first_batch_cache_miss_ratio
                else:
                    assert max_sim_steps == 1, "Not Implemented when max_sim_steps >= 2"
                    first_batch_token_ids_set = self._obtain_token_ids_set_from_sg_list(sg_list[:first_batch_num_seqs])
                    next_batch_token_ids_set = self._obtain_token_ids_set_from_sg_list(sg_list[first_batch_num_seqs:first_batch_num_seqs*2])
                    overlap_token_ids = next_batch_token_ids_set.intersection(first_batch_token_ids_set)
                    overlap_ratio = len(overlap_token_ids) / len(next_batch_token_ids_set)
                    subsequent_cache_miss_ratio = max(overlap_ratio, first_batch_cache_miss_ratio)

                # replace _pure_get_num_new_uncached_and_cached_tokens with cache_miss_ratio
                # as approximation for remaining requests
                next_batch_num_seqs = 0
                next_batch_uncached_tokens = 0
                next_batch_max_output_len = 0
                for sg in sg_list[first_batch_num_seqs:]:
                    estimated_num_new_tokens = subsequent_cache_miss_ratio * sg.get_seqs()[0].get_prompt_len()
                    if estimated_num_new_tokens + next_batch_uncached_tokens > self.scheduler_config.max_model_len:
                        # existing batch filled, calculate cost
                        total_cost += next_batch_uncached_tokens * slope_prefill + intercept_prefill
                        total_cost += next_batch_max_output_len * (next_batch_num_seqs * slope_decode
                                + intercept_decode)
                        # clear for new batch
                        next_batch_num_seqs = 0
                        next_batch_uncached_tokens = 0
                        next_batch_max_output_len = 0
                    # add request to batch
                    next_batch_num_seqs += 1
                    next_batch_uncached_tokens += estimated_num_new_tokens
                    next_batch_max_output_len = max(sg.sampling_params.max_tokens, next_batch_max_output_len)
                # deal with leftover ones
                if next_batch_num_seqs != 0:
                    total_cost += next_batch_uncached_tokens * slope_prefill + intercept_prefill
                    total_cost += next_batch_max_output_len * (next_batch_num_seqs * slope_decode
                            + intercept_decode)

            # update priority
            for sg in sg_list:
                sg.priority = total_cost
            record_relid_cost.append((rel_id, total_cost))

        new_waiting = []
        for rel_id, _ in sorted(record_relid_cost, key=lambda x:x[1]):
            new_waiting += waiting_queries[rel_id]
        self.waiting = deque(new_waiting)
        
        elapsed_time = time.perf_counter() - ts
        logger.info(f"priority updating overhead in seconds: {elapsed_time}")
     
    def _obtain_token_ids_set_from_sg_list(self, sg_list):
        all_prompt_tokens_set = set()
        for sg in sg_list:
            all_prompt_tokens_set.update(sg.get_seqs()[0].prompt_token_ids)
        return all_prompt_tokens_set

    def _construct_waiting_status(self):
        """
        status is a dict: rel_id ==> #reqs in that rQ
        """
        status = dict()
        for sg in self.waiting:
            if sg.rel_id not in status:
                status[sg.rel_id] = 0
            status[sg.rel_id] += 1
        return status

    def _starvation_prevention(self, original_waiting_relid_cost):
        """
        (1) update starvation threshold for new arrival relquery
        (2) check threshold and reset priority
        """
        agg_waiting = dict()
        for sg in self.waiting:
            if sg.rel_id not in agg_waiting:
                agg_waiting[sg.rel_id] = []
            agg_waiting[sg.rel_id].append(sg)

        for rid, sgs in agg_waiting.items():
            if rid not in self.starvation_threshold:
                # first arrival, calculate threshold and record arrival
                self.starvation_threshold[rid] = self.scheduler_config.starvation * len(sgs)
                self.relquery_arrival[rid] = min(sg.arrival_time for sg in sgs)

        starvation_count = 0
        old_waiting_priority_dict = {rid:cost for rid, cost in original_waiting_relid_cost}
        new_waiting_priority_dict = dict() # rid --> (cost, arrival_timestamp)
        for rid, sgs in agg_waiting.items():
            max_waiting_time = self.starvation_threshold[rid]
            arrival_ts = self.relquery_arrival[rid]
            current_waiting_time = time.time() - arrival_ts
            current_priority = old_waiting_priority_dict[rid]
            if current_waiting_time > max_waiting_time:
                # set priority as 0 to avoid starvation
                starvation_count += 1
                current_priority = 0
                logger.info(f"starvation of rid={rid}: max_waiting={max_waiting_time} s, current waiting={current_waiting_time} s")
            new_waiting_priority_dict[rid] = (current_priority, arrival_ts)

        logger.info(f"Total starvation count: {starvation_count}")

        # update self.waiting and record
        tmp_data = sorted(new_waiting_priority_dict.items(), key=lambda x:x[1])
        rectified_waiting_relid_cost = [(x[0], x[1][0]) for x in tmp_data]
        new_waiting = []
        for rel_id, _ in tmp_data:
            new_waiting += agg_waiting[rel_id]
        self.waiting = deque(new_waiting)
        return rectified_waiting_relid_cost

    def _check_reuse_update_waiting_priority_abs(self):
        """
        update (1) priority (2) content of self.waiting
        """
        agg_waiting = dict()
        for sg in self.waiting:
            if sg.rel_id not in agg_waiting:
                agg_waiting[sg.rel_id] = []
            agg_waiting[sg.rel_id].append(sg)
        new_waiting_queue_status = {rid:len(sgs) for rid, sgs in agg_waiting.items()}
        old_waiting_priority = {rid:cost for rid, cost in self.cached_waiting_priority} if self.cached_waiting_priority else None

        reuse_count = 0
        if new_waiting_queue_status != self.cached_waiting_queue_status:
            # check each relid and reuse if possible
            new_waiting_priority_dict = dict() # rid --> cost
            for rid, sgs in agg_waiting.items():
                if self.cached_waiting_queue_status is None or rid not in self.cached_waiting_queue_status or self.cached_waiting_queue_status[rid] != len(sgs):
                    # recompute for new or changed rQs
                    new_waiting_priority_dict[rid] = self._calc_single_query_priority(sgs)
                    #new_waiting_priority_dict[rid] = self._calc_single_query_priority_cd(sgs)
                else:
                    # reuse
                    reuse_count += 1
                    new_waiting_priority_dict[rid] = old_waiting_priority[rid]

            # update cache
            self.cached_waiting_queue_status = new_waiting_queue_status
            waiting_relid_cost = sorted(new_waiting_priority_dict.items(), key=lambda x:x[1])
            self.cached_waiting_priority = waiting_relid_cost

            # update self.waiting
            new_waiting = []
            for rel_id, _ in waiting_relid_cost:
                new_waiting += agg_waiting[rel_id]
            self.waiting = deque(new_waiting)
            logger.info(f"reuse {reuse_count} rQs' priority out of {len(new_waiting_priority_dict)}")
        else:
            logger.info(f"fully reuse waiting priority")

        return self.cached_waiting_priority
        

    def _maybe_update_waiting_priority_abs(self):
        """
        if waiting queue is not changed, return cached results to reduce overhead
        """
        waiting_status = self._construct_waiting_status()
        if waiting_status != self.cached_waiting_queue_status:
            # recompute 
            waiting_relid_cost = self._update_waiting_priority_abs()
            # update for future reuse
            self.cached_waiting_queue_status = waiting_status
            self.cached_waiting_priority = waiting_relid_cost
        else:
            logger.info("reuse waiting priority")

        return self.cached_waiting_priority

    def _schedule_default(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        The current policy is designed to optimize the throughput. First,
        it batches as many prefill requests as possible. And it schedules
        decodes. If there's a pressure on GPU memory, decode requests can
        be swapped or preempted.

        first determine prefill, then determine decode
        """
        if self.scheduler_config.policy in ['fcfs', 'priority']:
            return self._schedule_default_old()

        logger.info("%"*30+"xxxxxx"+"%"*30)
        assert self.scheduler_config.policy in ['priority_tc', 'priority_bs', 'priority_abs', 'priority_pabs', 
                'priority_ipabs', 'priority_iso_pabs', 'priority_ada_pabs', 'priority_islt_pabs', 'priority_ovlp_pabs']
        assert not self.lora_enabled
        assert not self.swapped
        curr_loras = None

        if self.scheduler_config.policy in ["priority_abs", "priority_pabs", "priority_iso_pabs", "priority_ipabs", 
                "priority_ada_pabs", "priority_islt_pabs", "priority_ovlp_pabs"]:
            # Include running requests to the budget.
            budget = SchedulingBudget(
                token_budget=self.scheduler_config.max_num_batched_tokens,
                max_num_seqs=self.scheduler_config.max_num_seqs,
            )

            # Update the priority for waiting & running relQueries
            assert len(self.scheduler_config.info_prefill) == 2
            assert len(self.scheduler_config.info_decode) == 2
            tic = time.perf_counter()
            #waiting_relid_cost = self._maybe_update_waiting_priority_abs()
            waiting_relid_cost = self._check_reuse_update_waiting_priority_abs()
            if self.scheduler_config.policy in ["priority_ada_pabs", "priority_islt_pabs"]:
                running_relid_cost = self._update_running_priority_abs(waiting_relid_cost)
            else:
                running_relid_cost = []
            if self.scheduler_config.policy in ["priority_ada_pabs"]:
                # avoid starvation
                waiting_relid_cost = self._starvation_prevention(waiting_relid_cost)
            elapsed_time = time.perf_counter() - tic
            logger.info(f"priority updating overhead in seconds: {elapsed_time}, waiting_relid_cost: {waiting_relid_cost}, running_relid_cost: {running_relid_cost}")

            # Make sure we include num running seqs before scheduling prefill,
            # so that we don't schedule beyond max_num_seqs for prefill.
            for seq_group in self.running:
                budget.add_num_seqs(seq_group.request_id,
                                    seq_group.get_max_num_running_seqs())

            prefills = SchedulerPrefillOutputs.create_empty()
            running_scheduled = SchedulerRunningOutputs.create_empty()
            swapped_in = SchedulerSwappedInOutputs.create_empty()

            # prefill
            if self.scheduler_config.policy in ["priority_ada_pabs", "priority_ovlp_pabs", "priority_islt_pabs"]:
                overlap_policy = self.scheduler_config.policy.split("_")[1]
                prefills = self._schedule_prefills_ada_pabs(budget,
                        waiting_relid_cost,
                        running_relid_cost,
                        overlap_policy,
                        curr_loras=None,
                        enable_chunking=False)
            else:
                prefill_sched_function = {
                        'priority_abs':self._schedule_prefills,
                        'priority_pabs':self._schedule_prefills_pabs, 
                        'priority_iso_pabs':self._schedule_prefills_iso_pabs,
                        'priority_ipabs':self._schedule_prefills_ipabs,
                        }
                prefills = prefill_sched_function[self.scheduler_config.policy](budget,
                        curr_loras=None,
                        enable_chunking=False)

        elif self.scheduler_config.policy == "priority_tc" :
            # Include running requests to the budget.
            budget = SchedulingBudget(
                token_budget=self.scheduler_config.max_num_batched_tokens,
                max_num_seqs=self.scheduler_config.max_num_seqs,
            )

            # update waiting requests' priority
            self._update_query_priority_tc()

            # Make sure we include num running seqs before scheduling prefill,
            # so that we don't schedule beyond max_num_seqs for prefill.
            for seq_group in self.running:
                budget.add_num_seqs(seq_group.request_id,
                                    seq_group.get_max_num_running_seqs())

            prefills = SchedulerPrefillOutputs.create_empty()
            running_scheduled = SchedulerRunningOutputs.create_empty()
            swapped_in = SchedulerSwappedInOutputs.create_empty()

            # prefill without preemption
            #prefills = self._schedule_prefills_separated(budget,
            prefills = self._schedule_prefills(budget,
                                               curr_loras,
                                               enable_chunking=False)

        elif self.scheduler_config.policy == "priority_bs":
            # Include running requests to the budget.
            budget = SchedulingBudget(
                token_budget=self.scheduler_config.max_num_batched_tokens,
                max_num_seqs=self.scheduler_config.max_num_seqs,
            )

            # update waiting requests' priority
            assert len(self.scheduler_config.info_prefill) == 2
            assert len(self.scheduler_config.info_decode) == 2
            self._update_query_priority_bs(1)

            # Make sure we include num running seqs before scheduling prefill,
            # so that we don't schedule beyond max_num_seqs for prefill.
            for seq_group in self.running:
                budget.add_num_seqs(seq_group.request_id,
                                    seq_group.get_max_num_running_seqs())

            prefills = SchedulerPrefillOutputs.create_empty()
            running_scheduled = SchedulerRunningOutputs.create_empty()
            swapped_in = SchedulerSwappedInOutputs.create_empty()

            # prefill without preemption
            prefills = self._schedule_prefills(budget,
                                               curr_loras,
                                               enable_chunking=False)
        else:
            raise ValueError

        # Don't schedule decodes if prefills are scheduled.
        # NOTE: If `_schedule_prefills` doesn't enable chunking, self.running
        # only contains decode requests, not chunked prefills.
        if len(prefills.seq_groups) == 0:
            logger.info(f"decode step")
            running_scheduled = self._schedule_running(budget,
                                                       curr_loras,
                                                       enable_chunking=False)

            # If any sequence group is preempted, do not swap in any sequence
            # group. because it means there's no slot for new running requests.
            if len(running_scheduled.preempted) + len(
                    running_scheduled.swapped_out) == 0:
                swapped_in = self._schedule_swapped(budget, curr_loras)
        else:
            logger.info(f"prefill step: {len(prefills.seq_groups)}")

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        if len(prefills.seq_groups) > 0:
            self.running.extend([s.seq_group for s in prefills.seq_groups])

        self.running.extend(running_scheduled.decode_seq_groups_list)

        if len(swapped_in.decode_seq_groups) > 0:
            self.running.extend(
                [s.seq_group for s in swapped_in.decode_seq_groups])

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)
        preempted = (len(running_scheduled.preempted) +
                     len(running_scheduled.swapped_out))

        # There should be no prefill from running queue because this policy
        # doesn't allow chunked prefills.
        assert len(running_scheduled.prefill_seq_groups) == 0
        assert len(swapped_in.prefill_seq_groups) == 0

        # Merge lists
        num_prefill_groups = len(prefills.seq_groups)
        if num_prefill_groups > 0:
            scheduled_seq_groups = prefills.seq_groups
            scheduled_seq_groups.extend(running_scheduled.decode_seq_groups)
        else:
            scheduled_seq_groups = running_scheduled.decode_seq_groups
        scheduled_seq_groups.extend(swapped_in.decode_seq_groups)

        blocks_to_copy = running_scheduled.blocks_to_copy
        blocks_to_copy.extend(swapped_in.blocks_to_copy)

        ignored_seq_groups = prefills.ignored_seq_groups
        ignored_seq_groups.extend(swapped_in.infeasible_seq_groups)

        logger.info(f"prefill={num_prefill_groups}, running={len(self.running)}, waiting={len(self.waiting)}")
        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens +
            budget.num_cached_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=preempted,
            num_uncached_prefill_tokens=budget.num_batched_tokens,
        )


    def _schedule_default_old(self) -> SchedulerOutputs:
        """Schedule queued requests.
        weird [priority + partial preempt] implementation of vllm
        
        The current policy is designed to optimize the throughput. First,
        it batches as many prefill requests as possible. And it schedules
        decodes. If there's a pressure on GPU memory, decode requests can
        be swapped or preempted.
        """
        # Include running requests to the budget.
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        # Make sure we include num running seqs before scheduling prefill,
        # so that we don't schedule beyond max_num_seqs for prefill.
        for seq_group in self.running:
            budget.add_num_seqs(seq_group.request_id,
                                seq_group.get_max_num_running_seqs())
        curr_loras = set(
            seq_group.lora_int_id for seq_group in self.running
            if seq_group.lora_int_id > 0) if self.lora_enabled else None

        prefills = SchedulerPrefillOutputs.create_empty()
        running_scheduled = SchedulerRunningOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        # If any requests are swapped, prioritized swapped requests.
        if not self.swapped:
            #prefills = self._schedule_prefills_separated(budget,
            prefills = self._schedule_prefills(budget,
                                               curr_loras,
                                               enable_chunking=False)

        if len(prefills.seq_groups
               ) == 0 and self.scheduler_config.policy == "priority":
            self._schedule_priority_preemption(budget)

        # Don't schedule decodes if prefills are scheduled.
        # NOTE: If `_schedule_prefills` doesn't enable chunking, self.running
        # only contains decode requests, not chunked prefills.
        if len(prefills.seq_groups) == 0:
            running_scheduled = self._schedule_running(budget,
                                                       curr_loras,
                                                       enable_chunking=False)

            # If any sequence group is preempted, do not swap in any sequence
            # group. because it means there's no slot for new running requests.
            if len(running_scheduled.preempted) + len(
                    running_scheduled.swapped_out) == 0:
                swapped_in = self._schedule_swapped(budget, curr_loras)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        if len(prefills.seq_groups) > 0:
            self.running.extend([s.seq_group for s in prefills.seq_groups])

        self.running.extend(running_scheduled.decode_seq_groups_list)

        if len(swapped_in.decode_seq_groups) > 0:
            self.running.extend(
                [s.seq_group for s in swapped_in.decode_seq_groups])

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)
        preempted = (len(running_scheduled.preempted) +
                     len(running_scheduled.swapped_out))

        # There should be no prefill from running queue because this policy
        # doesn't allow chunked prefills.
        assert len(running_scheduled.prefill_seq_groups) == 0
        assert len(swapped_in.prefill_seq_groups) == 0

        # Merge lists
        num_prefill_groups = len(prefills.seq_groups)
        if num_prefill_groups > 0:
            scheduled_seq_groups = prefills.seq_groups
            scheduled_seq_groups.extend(running_scheduled.decode_seq_groups)
        else:
            scheduled_seq_groups = running_scheduled.decode_seq_groups
        scheduled_seq_groups.extend(swapped_in.decode_seq_groups)

        blocks_to_copy = running_scheduled.blocks_to_copy
        blocks_to_copy.extend(swapped_in.blocks_to_copy)

        ignored_seq_groups = prefills.ignored_seq_groups
        ignored_seq_groups.extend(swapped_in.infeasible_seq_groups)

        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens +
            budget.num_cached_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=preempted,
            num_uncached_prefill_tokens=budget.num_batched_tokens,
        )

    def _schedule_chunked_prefill(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to be blocked
        by prefill requests.
        """
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        prefills = SchedulerPrefillOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        # Decoding should be always scheduled first by fcfs.
        running_scheduled = self._schedule_running(budget,
                                                   curr_loras,
                                                   enable_chunking=True)

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) == 0:
            swapped_in = self._schedule_swapped(budget, curr_loras)

        prefills = self._schedule_prefills(budget,
                                           curr_loras,
                                           enable_chunking=True)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)

        # Update new running requests.
        # By default, vLLM scheduler prioritizes prefills.
        # Once chunked prefill is enabled,
        # the policy is changed to prioritize decode requests.
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.prefill_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend([s.seq_group for s in prefills.seq_groups])

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)
        # Put prefills first due to Attention backend ordering assumption.
        scheduled_seq_groups = (prefills.seq_groups +
                                running_scheduled.prefill_seq_groups +
                                swapped_in.prefill_seq_groups +
                                running_scheduled.decode_seq_groups +
                                swapped_in.decode_seq_groups)
        num_prefill_groups = (len(prefills.seq_groups) +
                              len(swapped_in.prefill_seq_groups) +
                              len(running_scheduled.prefill_seq_groups))
        # If all prompts, then we set num_lookahead_slots to 0
        # this allows us to go through the `no_spec` path in
        # `spec_decode_worker.py`
        all_prefills = (len(scheduled_seq_groups) == num_prefill_groups)
        num_lookahead_slots = (0 if
                               (all_prefills
                                and not self.scheduler_config.is_multi_step)
                               else running_scheduled.num_lookahead_slots)
        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens +
            budget.num_cached_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
            swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups +
            swapped_in.infeasible_seq_groups,
            num_lookahead_slots=num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                       len(running_scheduled.swapped_out)),
        )

    def _schedule(self) -> SchedulerOutputs:
        """Schedule queued requests."""
        if self.scheduler_config.chunked_prefill_enabled:
            return self._schedule_chunked_prefill()
        else:
            return self._schedule_default()

    def _can_append_slots(self, seq_group: SequenceGroup,
                          enable_chunking: bool) -> bool:
        """Determine whether or not we have enough space in the KV cache to
        continue generation of the sequence group.
        """
        # It is True only for testing case to trigger artificial preemption.
        if (self.enable_artificial_preemption
                and random.uniform(0, 1) < ARTIFICIAL_PREEMPTION_PROB
                and self.artificial_preempt_cnt > 0):
            self.artificial_preempt_cnt -= 1
            return False

        is_prefill = seq_group.is_prefill()
        num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill, enable_chunking)

        if is_prefill and num_lookahead_slots > 0:
            # Appending prefill slots only happens multi-step and
            # chunked-prefill are enabled together.
            assert self.scheduler_config.is_multi_step and enable_chunking

        return self.block_manager.can_append_slots(
            seq_group=seq_group, num_lookahead_slots=num_lookahead_slots)

    def _allow_async_output_proc(self, seq_group: SequenceGroup) -> bool:
        # async_output_proc is allowed only when we have a single sequence
        # in the sequence group
        no_single_seq = seq_group.sampling_params is None or (
            seq_group.sampling_params.n == 1)
        return no_single_seq

    def schedule(
            self
    ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, bool]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_start_time = time.perf_counter()

        scheduler_outputs: SchedulerOutputs = self._schedule()
        now = time.time()

        if not self.cache_config.enable_prefix_caching:
            common_computed_block_nums = []

        allow_async_output_proc: bool = self.use_async_output_proc

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(
                scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            seq_group_metadata = self._seq_group_metadata_cache[
                self.cache_id].get_object()
            seq_group_metadata.seq_data.clear()
            seq_group_metadata.block_tables.clear()

            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}

            if seq_group.is_encoder_decoder():
                # Encoder associated with SequenceGroup
                encoder_seq = seq_group.get_encoder_seq()
                assert encoder_seq is not None
                encoder_seq_data = encoder_seq.data
                # Block table for cross-attention
                # Also managed at SequenceGroup level
                cross_block_table = self.block_manager.get_cross_block_table(
                    seq_group)
            else:
                encoder_seq_data = None
                cross_block_table = None

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)

            if self.cache_config.enable_prefix_caching:
                common_computed_block_nums = (
                    self.block_manager.get_common_computed_block_ids(
                        seq_group.get_seqs(status=SequenceStatus.RUNNING)))

            do_sample = True
            is_prompt = seq_group.is_prefill()
            # We should send the metadata to workers when the first prefill
            # is sent. Subsequent requests could be chunked prefill or decode.
            is_first_prefill = False
            if is_prompt:
                seqs = seq_group.get_seqs()
                # Prefill has only 1 sequence.
                assert len(seqs) == 1
                num_computed_tokens = seqs[0].data.get_num_computed_tokens()
                is_first_prefill = num_computed_tokens == 0
                # In the next iteration, all prompt tokens are not computed.
                # It means the prefill is chunked, and we don't need sampling.
                # NOTE: We use get_len instead of get_prompt_len because when
                # a sequence is preempted, prefill includes previous generated
                # output tokens.
                if (token_chunk_size + num_computed_tokens <
                        seqs[0].data.get_len()):
                    do_sample = False

            # It assumes the scheduled_seq_groups is ordered by
            # prefill < decoding.
            if is_first_prefill or not self.scheduler_config.send_delta_data:
                seq_group_metadata = SequenceGroupMetadata(
                    request_id=seq_group.request_id,
                    is_prompt=is_prompt,
                    seq_data=seq_data,
                    sampling_params=seq_group.sampling_params,
                    block_tables=block_tables,
                    do_sample=do_sample,
                    pooling_params=seq_group.pooling_params,
                    token_chunk_size=token_chunk_size,
                    lora_request=seq_group.lora_request,
                    computed_block_nums=common_computed_block_nums,
                    encoder_seq_data=encoder_seq_data,
                    cross_block_table=cross_block_table,
                    state=seq_group.state,
                    token_type_ids=seq_group.token_type_ids,
                    # `multi_modal_data` will only be present for the 1st comm
                    # between engine and worker.
                    # the subsequent comms can still use delta, but
                    # `multi_modal_data` will be None.
                    multi_modal_data=seq_group.multi_modal_data
                    if scheduler_outputs.num_prefill_groups > 0 else None,
                    multi_modal_placeholders=seq_group.multi_modal_placeholders
                    if scheduler_outputs.num_prefill_groups > 0 else None,
                    mm_processor_kwargs=seq_group.mm_processor_kwargs,
                    prompt_adapter_request=seq_group.prompt_adapter_request,
                )
            else:
                # When SPMD mode is enabled, we only send delta data except for
                # the first request to reduce serialization cost.
                seq_data_delta = {}
                for id, data in seq_data.items():
                    seq_data_delta[id] = data.get_delta_and_reset()
                seq_group_metadata = SequenceGroupMetadataDelta(
                    seq_data_delta,
                    seq_group.request_id,
                    block_tables,
                    is_prompt,
                    do_sample=do_sample,
                    token_chunk_size=token_chunk_size,
                    computed_block_nums=common_computed_block_nums,
                )
            seq_group_metadata_list.append(seq_group_metadata)

            if allow_async_output_proc:
                allow_async_output_proc = self._allow_async_output_proc(
                    seq_group)

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group,
                scheduled_seq_group.token_chunk_size)

        self._seq_group_metadata_cache[self.next_cache_id].reset()

        scheduler_time = time.perf_counter() - scheduler_start_time
        # Add this to scheduler time to all the sequences that are currently
        # running. This will help estimate if the scheduler is a significant
        # component in the e2e latency.
        for seq_group in self.running:
            if seq_group is not None and seq_group.metrics is not None:
                if seq_group.metrics.scheduler_time is not None:
                    seq_group.metrics.scheduler_time += scheduler_time
                else:
                    seq_group.metrics.scheduler_time = scheduler_time

        # Move to next cache (if exists)
        self.cache_id = self.next_cache_id

        # Return results
        return (seq_group_metadata_list, scheduler_outputs,
                allow_async_output_proc)

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        """Free a sequence from a block table."""
        self.block_manager.free(seq)

    def _free_finished_seqs(self, seq_group: SequenceGroup) -> None:
        """Free finished seqs in a sequence group."""
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                self.free_seq(seq)

    def _free_finished_seq_group(self, seq_group: SequenceGroup) -> None:
        if seq_group.is_finished():
            # Free cross-attention block table, if it exists
            self._free_seq_group_cross_attn_blocks(seq_group)

            # Add the finished requests to the finished requests list.
            # This list will be used to update the Mamba cache in the
            # next step.
            self._finished_requests_ids.append(seq_group.request_id)

        # Free finished seqs
        self._free_finished_seqs(seq_group)

    def free_finished_seq_groups(self) -> None:
        remaining: Deque[SequenceGroup] = deque()
        for seq_group in self.running:
            self._free_finished_seq_group(seq_group)
            if not seq_group.is_finished():
                remaining.append(seq_group)

        self.running = remaining

        # Handle async stopped sequence groups
        # (ones that reached max model len)
        if self._async_stopped:
            for seq_group in self._async_stopped:
                self._free_seq_group_cross_attn_blocks(seq_group)
                self._finished_requests_ids.append(seq_group.request_id)

                # Free finished seqs
                self._free_finished_seqs(seq_group)

            self._async_stopped.clear()

    def _allocate_and_set_running(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slots(self,
                      seq_group: SequenceGroup,
                      blocks_to_copy: List[Tuple[int, int]],
                      enable_chunking: bool = False) -> None:
        """Appends new slots to the sequences in the given sequence group.

        Args:
            seq_group (SequenceGroup): The sequence group containing the
                sequences to append slots to.
            blocks_to_copy (List[Tuple[int, int]]): A list of tuple of two
                ints, the first int is the source block index, and the second
                int is the destination block index. This list is updated with
                the new source and destination block indices for the appended
                slots.
            enable_chunking (bool): True if chunked prefill is enabled.
        """
        is_prefill: bool = seq_group.is_prefill()
        num_lookahead_slots: int = self._get_num_lookahead_slots(
            is_prefill, enable_chunking)

        seq_group.init_multi_step_from_lookahead_slots(
            num_lookahead_slots,
            num_scheduler_steps=self.scheduler_config.num_scheduler_steps,
            is_multi_step=self.scheduler_config.is_multi_step,
            enable_chunking=enable_chunking)

        seq_status: Optional[SequenceStatus] = SequenceStatus.RUNNING
        if self.scheduler_config.is_multi_step and enable_chunking:
            # In multi-step chunked-prefill any sequence type can have
            # slots appended.
            seq_status = None

        for seq in seq_group.get_seqs(status=seq_status):
            cows = self.block_manager.append_slots(seq, num_lookahead_slots)
            if len(cows) > 0:
                blocks_to_copy.extend(cows)

    def _preempt(self, seq_group: SequenceGroup,
                 blocks_to_swap_out: List[Tuple[int, int]]) -> PreemptionMode:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if self.user_specified_preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP

        elif self.user_specified_preemption_mode == "swap":
            preemption_mode = PreemptionMode.SWAP
        else:
            preemption_mode = PreemptionMode.RECOMPUTE

        if self.num_cumulative_preemption % 10 == 0:
            logger.warning(
                "Sequence group %s is preempted by %s mode because there is "
                "not enough KV cache space. This can affect the end-to-end "
                "performance. Increase gpu_memory_utilization or "
                "tensor_parallel_size to provide more KV cache memory. "
                "total_num_cumulative_preemption=%d", seq_group.request_id,
                preemption_mode, self.num_cumulative_preemption + 1)
        self.num_cumulative_preemption += 1

        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")
        return preemption_mode

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.free_seq(seq)
            seq.reset_state_for_recompute()

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: List[Tuple[int, int]],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

    def _passed_delay(self, now: float) -> bool:
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min(
                [e.metrics.arrival_time for e in self.waiting])
            passed_delay = (
                (now - earliest_arrival_time) >
                (self.scheduler_config.delay_factor * self.last_prompt_latency)
                or not self.running)
        else:
            passed_delay = True
        return passed_delay

    def _get_num_lookahead_slots(self, is_prefill: bool,
                                 enable_chunking: bool) -> int:
        """The number of slots to allocate per sequence per step, beyond known
        token ids. Speculative decoding uses these slots to store KV activations
        of tokens which may or may not be accepted.

        Speculative decoding does not yet support prefill, so we do not perform
        lookahead allocation for prefill.

        When chunking is enabled with multi-step, we allocate lookahead slots
        for the prefills for when the prefills turn into decodes in the first
        step.
        """
        if is_prefill:
            if self.scheduler_config.is_multi_step and enable_chunking:
                # num_lookahead_slots was introduced in the context of decodes,
                # in Speculative Decoding.
                # When the num_scheduler_steps is 8, say, then the
                # num_lookahead_slots is 7. Meaning, we are doing a 1-step of
                # decode anyways and we wish to do 7 more.
                #
                # "lookaheads" for prefills, is introduced in support for
                # Chunked-Prefill in Multi-Step.
                return self.scheduler_config.num_lookahead_slots + 1
            else:
                return 0

        return self.scheduler_config.num_lookahead_slots

    def _pure_get_num_new_uncached_and_cached_tokens(
        self,
        seq_group: SequenceGroup,
        status: SequenceStatus,
    ) -> Tuple[int, int]:
        """
        Returns the number of new uncached and cached tokens to schedule for a
        given sequence group that's in a given `status`.

        budget and chunking are not considered

        Args:
            seq_group: The sequence group to get the number of new tokens to
                schedule.
            status: The status of the sequences to get the number of new tokens
                to schedule.

        Returns:
            A tuple of two ints. The first int is the number of new uncached
            tokens to schedule. The second int is the number of cached tokens.
        """
        num_cached_new_tokens = 0
        num_uncached_new_tokens = 0

        seqs = seq_group.get_seqs(status=status)
        # Compute the number of new uncached and cached tokens for
        # each sequence.
        for seq in seqs:
            if not seq.is_prefill():
                # Decode sequences should always just have 1 uncached token
                # TODO(rickyx): Actually is this still correct for multi-step?
                num_uncached_new_tokens += 1
                continue

            num_computed_tokens_seq = seq.get_num_computed_tokens()
            all_num_new_tokens_seq = seq.get_len() - num_computed_tokens_seq
            if not self.cache_config.enable_prefix_caching:
                # If prefix caching is not enabled, all new tokens are uncached.
                num_uncached_new_tokens += all_num_new_tokens_seq
                continue

            # NOTE: the cache token might be currently in a block that's in an
            # evictor meaning that it's not yet allocated. However, we don't
            # exclude such tokens in the cache count because it will be
            # guaranteed to be allocated later if the sequence can be allocated.
            num_cached_tokens_seq = self.block_manager.get_num_cached_tokens(
                seq)

            # Sanity check.
            if num_cached_tokens_seq < num_computed_tokens_seq:
                # This should only happen with chunked prefill, and
                # the seq is still in prefill. The `num_cached_tokens_seq`
                # is the value we calculated on scheduling the first prefill.
                # For subsequent continuous prefill steps, we cached the
                # number of cache tokens for the sequence so the cached token
                # count could be less than the number of computed tokens.
                # See comments on `ComputedBlocksTracker` for more details.
                assert (
                    seq.is_prefill() and seq.status == SequenceStatus.RUNNING
                    and self.scheduler_config.chunked_prefill_enabled
                ), ("Number of cached tokens should not be less than the "
                    "number of computed tokens for a sequence that's still "
                    f"in prefill. But there are {num_cached_tokens_seq} cached "
                    f"tokens and {num_computed_tokens_seq} computed tokens "
                    f"for sequence {seq.seq_id}.")

            num_cached_new_tokens_seq = max(
                0, num_cached_tokens_seq - num_computed_tokens_seq)
            num_uncached_new_tokens_seq = (all_num_new_tokens_seq -
                                           num_cached_new_tokens_seq)

            num_uncached_new_tokens += num_uncached_new_tokens_seq
            num_cached_new_tokens += num_cached_new_tokens_seq

        if num_uncached_new_tokens == 0 and num_cached_new_tokens > 0:
            # For a fully cached hit sequence, we actually need to recompute the
            # last token. So we need at least 1 uncached token to schedule.
            # See ModelRunner._compute_for_prefix_cache_hit for more details.
            num_uncached_new_tokens = 1
            num_cached_new_tokens -= 1

        return num_uncached_new_tokens, num_cached_new_tokens


    def _get_num_new_uncached_and_cached_tokens(
        self,
        seq_group: SequenceGroup,
        status: SequenceStatus,
        enable_chunking: bool,
        budget: SchedulingBudget,
    ) -> Tuple[int, int]:
        """
        Returns the number of new uncached and cached tokens to schedule for a
        given sequence group that's in a given `status`.

        The API could chunk the number of tokens to compute based on `budget`
        if `enable_chunking` is True. If a sequence group has multiple
        sequences (e.g., running beam search), it means it is in decoding
        phase, so chunking doesn't happen.

        Returns (0, 0) if the new token cannot be computed due to token budget.

        The cached tokens's blocks are already computed, and the attention
        backend will reuse the cached blocks rather than recomputing them. So
        the scheduler could schedule these cached tokens "for free".

        Args:
            seq_group: The sequence group to get the number of new tokens to
                schedule.
            status: The status of the sequences to get the number of new tokens
                to schedule.
            enable_chunking: Whether to chunk the number of tokens to compute.
            budget: The budget to chunk the number of tokens to compute.


        Returns:
            A tuple of two ints. The first int is the number of new uncached
            tokens to schedule. The second int is the number of cached tokens.
            If no more new tokens can be scheduled, returns (0, 0).
        """
        num_cached_new_tokens = 0
        num_uncached_new_tokens = 0

        seqs = seq_group.get_seqs(status=status)
        # Compute the number of new uncached and cached tokens for
        # each sequence.
        for seq in seqs:
            if not seq.is_prefill():
                # Decode sequences should always just have 1 uncached token
                # TODO(rickyx): Actually is this still correct for multi-step?
                num_uncached_new_tokens += 1
                continue

            num_computed_tokens_seq = seq.get_num_computed_tokens()
            all_num_new_tokens_seq = seq.get_len() - num_computed_tokens_seq
            if not self.cache_config.enable_prefix_caching:
                # If prefix caching is not enabled, all new tokens are uncached.
                num_uncached_new_tokens += all_num_new_tokens_seq
                continue

            # NOTE: the cache token might be currently in a block that's in an
            # evictor meaning that it's not yet allocated. However, we don't
            # exclude such tokens in the cache count because it will be
            # guaranteed to be allocated later if the sequence can be allocated.
            num_cached_tokens_seq = self.block_manager.get_num_cached_tokens(
                seq)

            # Sanity check.
            if num_cached_tokens_seq < num_computed_tokens_seq:
                # This should only happen with chunked prefill, and
                # the seq is still in prefill. The `num_cached_tokens_seq`
                # is the value we calculated on scheduling the first prefill.
                # For subsequent continuous prefill steps, we cached the
                # number of cache tokens for the sequence so the cached token
                # count could be less than the number of computed tokens.
                # See comments on `ComputedBlocksTracker` for more details.
                assert (
                    seq.is_prefill() and seq.status == SequenceStatus.RUNNING
                    and self.scheduler_config.chunked_prefill_enabled
                ), ("Number of cached tokens should not be less than the "
                    "number of computed tokens for a sequence that's still "
                    f"in prefill. But there are {num_cached_tokens_seq} cached "
                    f"tokens and {num_computed_tokens_seq} computed tokens "
                    f"for sequence {seq.seq_id}.")

            num_cached_new_tokens_seq = max(
                0, num_cached_tokens_seq - num_computed_tokens_seq)
            num_uncached_new_tokens_seq = (all_num_new_tokens_seq -
                                           num_cached_new_tokens_seq)

            num_uncached_new_tokens += num_uncached_new_tokens_seq
            num_cached_new_tokens += num_cached_new_tokens_seq

        if num_uncached_new_tokens == 0 and num_cached_new_tokens > 0:
            # For a fully cached hit sequence, we actually need to recompute the
            # last token. So we need at least 1 uncached token to schedule.
            # See ModelRunner._compute_for_prefix_cache_hit for more details.
            num_uncached_new_tokens = 1
            num_cached_new_tokens -= 1

        if enable_chunking and len(seqs) == 1:
            # Chunk if a running request cannot fit in the given budget.
            # If number of seq > 1, it means it is doing beam search
            # in a decode phase. Do not chunk.
            num_uncached_new_tokens = self._chunk_new_tokens_to_schedule(
                self.scheduler_config,
                self.cache_config,
                budget,
                self._get_prompt_limit(seq_group),
                num_uncached_new_tokens,
            )

        return num_uncached_new_tokens, num_cached_new_tokens

    @staticmethod
    def _chunk_new_tokens_to_schedule(
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        budget: SchedulingBudget,
        prompt_limit: int,
        num_new_tokens: int,
    ) -> int:
        """
        Chunks the number of new tokens to schedule based on the budget when
        chunked prefill is enabled.

        Args:
            scheduler_config: The scheduler config.
            cache_config: The cache config.
            budget: The budget to chunk the number of tokens to compute.
            prompt_limit: The maximum number of tokens allowed in a prompt.
            num_new_tokens: The number of new tokens to schedule.

        Returns:
            The number of new tokens to schedule after chunking.
        """
        remaining_token_budget = budget.remaining_token_budget()
        if scheduler_config.is_multi_step:
            # The current multi-step + chunked prefill capability does
            # not actually support chunking prompts.
            #
            # Therefore, `num_new_tokens` is computed in the same fashion
            # for both multi-step+chunked-prefill &
            # multi-step+chunked-prefill+APC
            #
            # Prompts with more tokens than the current remaining budget
            # are postponed to future scheduler steps
            if num_new_tokens > prompt_limit:
                # If the seq_group is in prompt-stage, pass the
                # num_new_tokens as-is so the caller can ignore
                # the sequence.
                return num_new_tokens

            return (0 if num_new_tokens > remaining_token_budget else
                    num_new_tokens)

        if cache_config.enable_prefix_caching:
            # Adjust the remaining token budget to be divisible by the block
            # size when prefix caching is enabled.

            # When prefix caching is enabled, we always allocate
            # the number of new tokens that is dividable by the block
            # size to avoid partial block matching.
            block_size = cache_config.block_size
            remainder = budget.token_budget % block_size
            if remainder != 0:
                raise ValueError("When enabling chunked prefill and "
                                 "prefix caching, max_num_batched_tokens "
                                 "(chunk size) must be dividable by "
                                 "block size, but got chunk_size "
                                 f"({budget.token_budget}) % block_size "
                                 f"({block_size}) = {remainder}")
            # Round down to block size.
            remaining_token_budget = (remaining_token_budget // block_size *
                                      block_size)

        num_new_tokens = min(num_new_tokens, remaining_token_budget)

        return num_new_tokens
