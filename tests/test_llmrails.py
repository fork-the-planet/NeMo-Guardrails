# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Union

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.llmrails import _get_action_details_from_flow_id
from tests.utils import FakeLLM, clean_events, event_sequence_conforms


@pytest.fixture
def rails_config():
    return RailsConfig.parse_object(
        {
            "models": [
                {
                    "type": "main",
                    "engine": "fake",
                    "model": "fake",
                }
            ],
            "user_messages": {
                "express greeting": ["Hello!"],
                "ask math question": ["What is 2 + 2?", "5 + 9"],
            },
            "flows": [
                {
                    "elements": [
                        {"user": "express greeting"},
                        {"bot": "express greeting"},
                    ]
                },
                {
                    "elements": [
                        {"user": "ask math question"},
                        {"execute": "compute"},
                        {"bot": "provide math response"},
                        {"bot": "ask if user happy"},
                    ]
                },
            ],
            "bot_messages": {
                "express greeting": ["Hello! How are you?"],
                "provide response": ["The answer is 234", "The answer is 1412"],
            },
        }
    )


@pytest.mark.asyncio
async def test_1(rails_config):
    llm = FakeLLM(
        responses=[
            "  express greeting",
            "  ask math question",
            '  "The answer is 5"',
            '  "Are you happy with the result?"',
        ]
    )

    async def compute(context: dict, what: Optional[str] = "2 + 3"):
        return eval(what)

    llm_rails = LLMRails(config=rails_config, llm=llm)
    llm_rails.runtime.register_action(compute)

    events = [{"type": "UtteranceUserActionFinished", "final_transcript": "Hello!"}]

    new_events = await llm_rails.runtime.generate_events(events)
    clean_events(new_events)

    expected_events = [
        {
            "data": {"user_message": "Hello!"},
            "source_uid": "NeMoGuardrails",
            "type": "ContextUpdate",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "UserMessage", "text": "$user_message"}
            },
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "UserMessage", "text": "$user_message"}
            },
            "action_result_key": None,
            "events": [
                {
                    "source_uid": "NeMoGuardrails",
                    "text": "Hello!",
                    "type": "UserMessage",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "source_uid": "NeMoGuardrails",
            "text": "Hello!",
            "type": "UserMessage",
        },
        {
            "action_name": "generate_user_intent",
            "action_params": {},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "generate_user_intent",
            "action_params": {},
            "action_result_key": None,
            "events": [
                {
                    "intent": "express greeting",
                    "source_uid": "NeMoGuardrails",
                    "type": "UserIntent",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "intent": "express greeting",
            "source_uid": "NeMoGuardrails",
            "type": "UserIntent",
        },
        {
            "intent": "express greeting",
            "source_uid": "NeMoGuardrails",
            "type": "BotIntent",
        },
        {
            "action_name": "retrieve_relevant_chunks",
            "action_params": {},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "data": {"relevant_chunks": "\n"},
            "source_uid": "NeMoGuardrails",
            "type": "ContextUpdate",
        },
        {
            "action_name": "retrieve_relevant_chunks",
            "action_params": {},
            "action_result_key": None,
            "events": None,
            "is_success": True,
            "is_system_action": True,
            "return_value": "\n",
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "action_name": "generate_bot_message",
            "action_params": {},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "data": {"skip_output_rails": True},
            "source_uid": "NeMoGuardrails",
            "type": "ContextUpdate",
        },
        {
            "action_name": "generate_bot_message",
            "action_params": {},
            "action_result_key": None,
            "events": [
                {
                    "source_uid": "NeMoGuardrails",
                    "text": "Hello! How are you?",
                    "type": "BotMessage",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "source_uid": "NeMoGuardrails",
            "text": "Hello! How are you?",
            "type": "BotMessage",
        },
        {
            "data": {"bot_message": "Hello! How are you?", "skip_output_rails": False},
            "source_uid": "NeMoGuardrails",
            "type": "ContextUpdate",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "StartUtteranceBotAction", "script": "$bot_message"}
            },
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "StartUtteranceBotAction", "script": "$bot_message"}
            },
            "action_result_key": None,
            "events": [
                {
                    "action_info_modality": "bot_speech",
                    "action_info_modality_policy": "replace",
                    "script": "Hello! How are you?",
                    "source_uid": "NeMoGuardrails",
                    "type": "StartUtteranceBotAction",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "action_info_modality": "bot_speech",
            "action_info_modality_policy": "replace",
            "script": "Hello! How are you?",
            "source_uid": "NeMoGuardrails",
            "type": "StartUtteranceBotAction",
        },
        {
            "source_uid": "NeMoGuardrails",
            "type": "Listen",
        },
    ]

    # assert expected_events == new_events

    assert event_sequence_conforms(expected_events, new_events)

    events.extend(new_events)
    events.append({"type": "UtteranceUserActionFinished", "final_transcript": "2 + 3"})

    new_events = await llm_rails.runtime.generate_events(events)
    clean_events(new_events)

    expected_events = [
        {
            "data": {"user_message": "2 + 3"},
            "source_uid": "NeMoGuardrails",
            "type": "ContextUpdate",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "UserMessage", "text": "$user_message"}
            },
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "UserMessage", "text": "$user_message"}
            },
            "action_result_key": None,
            "events": [
                {
                    "source_uid": "NeMoGuardrails",
                    "text": "2 + 3",
                    "type": "UserMessage",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "source_uid": "NeMoGuardrails",
            "text": "2 + 3",
            "type": "UserMessage",
        },
        {
            "action_name": "generate_user_intent",
            "action_params": {},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "generate_user_intent",
            "action_params": {},
            "action_result_key": None,
            "events": [
                {
                    "intent": "ask math question",
                    "source_uid": "NeMoGuardrails",
                    "type": "UserIntent",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "intent": "ask math question",
            "source_uid": "NeMoGuardrails",
            "type": "UserIntent",
        },
        {
            "action_name": "compute",
            "action_params": {},
            "action_result_key": None,
            "is_system_action": False,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "compute",
            "action_params": {},
            "action_result_key": None,
            "events": [],
            "is_success": True,
            "is_system_action": False,
            "return_value": 5,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "intent": "provide math response",
            "source_uid": "NeMoGuardrails",
            "type": "BotIntent",
        },
        {
            "action_name": "retrieve_relevant_chunks",
            "action_params": {},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "data": {"relevant_chunks": "\n\n"},
            "source_uid": "NeMoGuardrails",
            "type": "ContextUpdate",
        },
        {
            "action_name": "retrieve_relevant_chunks",
            "action_params": {},
            "action_result_key": None,
            "events": None,
            "is_success": True,
            "is_system_action": True,
            "return_value": "\n\n",
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "action_name": "generate_bot_message",
            "action_params": {},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "generate_bot_message",
            "action_params": {},
            "action_result_key": None,
            "events": [
                {
                    "source_uid": "NeMoGuardrails",
                    "text": "The answer is 5",
                    "type": "BotMessage",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "source_uid": "NeMoGuardrails",
            "text": "The answer is 5",
            "type": "BotMessage",
        },
        {
            "data": {"bot_message": "The answer is 5"},
            "source_uid": "NeMoGuardrails",
            "type": "ContextUpdate",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "StartUtteranceBotAction", "script": "$bot_message"}
            },
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "StartUtteranceBotAction", "script": "$bot_message"}
            },
            "action_result_key": None,
            "events": [
                {
                    "action_info_modality": "bot_speech",
                    "action_info_modality_policy": "replace",
                    "script": "The answer is 5",
                    "source_uid": "NeMoGuardrails",
                    "type": "StartUtteranceBotAction",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "action_info_modality": "bot_speech",
            "action_info_modality_policy": "replace",
            "script": "The answer is 5",
            "source_uid": "NeMoGuardrails",
            "type": "StartUtteranceBotAction",
        },
        {
            "intent": "ask if user happy",
            "source_uid": "NeMoGuardrails",
            "type": "BotIntent",
        },
        {
            "action_name": "retrieve_relevant_chunks",
            "action_params": {},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "data": {"relevant_chunks": "\n\n\n"},
            "source_uid": "NeMoGuardrails",
            "type": "ContextUpdate",
        },
        {
            "action_name": "retrieve_relevant_chunks",
            "action_params": {},
            "action_result_key": None,
            "events": None,
            "is_success": True,
            "is_system_action": True,
            "return_value": "\n\n\n",
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "action_name": "generate_bot_message",
            "action_params": {},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "generate_bot_message",
            "action_params": {},
            "action_result_key": None,
            "events": [
                {
                    "source_uid": "NeMoGuardrails",
                    "text": "Are you happy with the result?",
                    "type": "BotMessage",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "source_uid": "NeMoGuardrails",
            "text": "Are you happy with the result?",
            "type": "BotMessage",
        },
        {
            "data": {"bot_message": "Are you happy with the result?"},
            "source_uid": "NeMoGuardrails",
            "type": "ContextUpdate",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "StartUtteranceBotAction", "script": "$bot_message"}
            },
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "StartUtteranceBotAction", "script": "$bot_message"}
            },
            "action_result_key": None,
            "events": [
                {
                    "action_info_modality": "bot_speech",
                    "action_info_modality_policy": "replace",
                    "script": "Are you happy with the result?",
                    "source_uid": "NeMoGuardrails",
                    "type": "StartUtteranceBotAction",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "action_info_modality": "bot_speech",
            "action_info_modality_policy": "replace",
            "script": "Are you happy with the result?",
            "source_uid": "NeMoGuardrails",
            "type": "StartUtteranceBotAction",
        },
        {
            "source_uid": "NeMoGuardrails",
            "type": "Listen",
        },
    ]

    # assert expected_events == new_events
    assert event_sequence_conforms(expected_events, new_events)


@pytest.mark.asyncio
async def test_2(rails_config):
    llm = FakeLLM(
        responses=[
            "  express greeting",
            "  ask math question",
            '  "The answer is 5"',
            '  "Are you happy with the result?"',
        ]
    )

    async def compute(what: Optional[str] = "2 + 3"):
        return eval(what)

    llm_rails = LLMRails(config=rails_config, llm=llm)
    llm_rails.runtime.register_action(compute)

    messages = [{"role": "user", "content": "Hello!"}]
    bot_message = await llm_rails.generate_async(messages=messages)

    assert bot_message == {"role": "assistant", "content": "Hello! How are you?"}
    messages.append(bot_message)

    messages.append({"role": "user", "content": "2 + 3"})
    bot_message = await llm_rails.generate_async(messages=messages)
    assert bot_message == {
        "role": "assistant",
        "content": "The answer is 5\nAre you happy with the result?",
    }


# get_action_details_from_flow_id used in llmrails.py


@pytest.fixture
def dummy_flows() -> List[Union[Dict, Any]]:
    return [
        {
            "id": "test_flow",
            "elements": [
                {
                    "_type": "run_action",
                    "_source_mapping": {
                        "filename": "flows.v1.co",
                        "line_text": "execute something",
                    },
                    "action_name": "test_action",
                    "action_params": {"param1": "value1"},
                }
            ],
        },
        # Additional flow that should match on a prefix
        {
            "id": "other_flow is prefix",
            "elements": [
                {
                    "_type": "run_action",
                    "_source_mapping": {
                        "filename": "flows.v1.co",
                        "line_text": "execute something else",
                    },
                    "action_name": "other_action",
                    "action_params": {"param2": "value2"},
                }
            ],
        },
        {
            "id": "test_rails_co",
            "elements": [
                {
                    "_type": "run_action",
                    "_source_mapping": {
                        "filename": "rails.co",
                        "line_text": "execute something",
                    },
                    "action_name": "test_action_supported",
                    "action_params": {"param1": "value1"},
                }
            ],
        },
        {
            "id": "test_rails_co_v2",
            "elements": [
                {
                    "_type": "run_action",
                    "_source_mapping": {
                        "filename": "rails.co",
                        "line_text": "await something",  # in colang 2 we use await
                    },
                    "action_name": "test_action_not_supported",
                    "action_params": {"param1": "value1"},
                }
            ],
        },
    ]


def test_get_action_details_exact_match(dummy_flows):
    action_name, action_params = _get_action_details_from_flow_id(
        "test_flow", dummy_flows
    )
    assert action_name == "test_action"
    assert action_params == {"param1": "value1"}


def test_get_action_details_exact_match_any_co_file(dummy_flows):
    action_name, action_params = _get_action_details_from_flow_id(
        "test_rails_co", dummy_flows
    )
    assert action_name == "test_action_supported"
    assert action_params == {"param1": "value1"}


def test_get_action_details_exact_match_not_colang_2(dummy_flows):
    with pytest.raises(ValueError) as exc_info:
        _get_action_details_from_flow_id("test_rails_co_v2", dummy_flows)

    assert "No run_action element found for flow_id" in str(exc_info.value)


def test_get_action_details_prefix_match(dummy_flows):
    # For a flow_id that starts with the prefix "other_flow",
    # we expect to retrieve the action details from the flow whose id starts with that prefix.
    # we expect a result since we are passing the prefixes argument.
    action_name, action_params = _get_action_details_from_flow_id(
        "other_flow", dummy_flows, prefixes=["other_flow"]
    )
    assert action_name == "other_action"
    assert action_params == {"param2": "value2"}


def test_get_action_details_prefix_match_unsupported_prefix(dummy_flows):
    # For a flow_id that starts with the prefix "other_flow",
    # we expect to retrieve the action details from the flow whose id starts with that prefix.
    # but as the prefix is not supported, we expect a ValueError.

    with pytest.raises(ValueError) as exc_info:
        _get_action_details_from_flow_id("other_flow", dummy_flows)

    assert "No action found for flow_id" in str(exc_info.value)


def test_get_action_details_no_match(dummy_flows):
    # Tests that a non matching flow_id raises a ValueError
    with pytest.raises(ValueError) as exc_info:
        _get_action_details_from_flow_id("non_existing_flow", dummy_flows)
    assert "No action found for flow_id" in str(exc_info.value)
