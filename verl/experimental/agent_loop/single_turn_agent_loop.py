# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            gen_output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = gen_output.num_preempted if gen_output.num_preempted is not None else -1
        response_mask = [1] * len(gen_output.token_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=gen_output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=gen_output.log_probs[: self.response_length] if gen_output.log_probs else None,
            response_entropies=gen_output.entropy[: self.response_length] if gen_output.entropy else None,
            response_variances=gen_output.variance[: self.response_length] if gen_output.variance else None,
            routed_experts=(
                gen_output.routed_experts[: len(prompt_ids) + self.response_length]
                if gen_output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output
