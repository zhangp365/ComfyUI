import ast
import asyncio
import copy
import json
import logging
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import uuid

from aiohttp import web
from aiohttp.multipart import MultipartWriter
import pytest

from tools.cache import ConnectCache


@pytest.fixture
def handlers(monkeypatch):
    source = Path(__file__).resolve().parents[2] / "server.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    names = {"post_prompt", "excute_sampling_task", "_get_history_response"}
    functions = [node for node in ast.walk(module) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    for function in functions:
        function.decorator_list = []
    queue = SimpleNamespace(get_current_queue=lambda: ([], []), put=Mock())
    server = SimpleNamespace(
        number=0,
        prompt_queue=queue,
        trigger_on_prompt=lambda data: data,
        node_replace_manager=SimpleNamespace(apply_replacements=Mock()),
    )
    execution = SimpleNamespace(
        validate_prompt=AsyncMock(return_value=(True, None, ["output"], {})),
        SENSITIVE_EXTRA_DATA_KEYS=("auth_token_comfy_org", "api_key_comfy_org"),
    )
    monkeypatch.setattr(ConnectCache, "cache", {})
    namespace = {
        "self": server, "execution": execution, "web": web,
        "logger": logging.getLogger(__name__), "logging": logging,
        "asyncio": asyncio, "json": json, "copy": copy, "time": time, "uuid": uuid,
        "ConnectCache": ConnectCache, "MultipartWriter": MultipartWriter,
        "process_multipart": AsyncMock(), "tensor2Bytes": lambda value: value,
        "valid_workflow_metadata": lambda data: None,
        "workflow_metadata_from_prompt": lambda data: None,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    return SimpleNamespace(namespace=namespace, server=server, execution=execution)


@pytest.mark.parametrize("route", ["post_prompt", "excute_sampling_task"])
def test_local_prompt_id_and_queue_contract(handlers, route):
    prompt = {"output": {"class_type": "SaveImage", "inputs": {}}}
    payload = {
        "prompt_id": "local-task-123", "prompt": prompt,
        "partial_execution_targets": ["output"],
        "extra_data": {"auth_token_comfy_org": "secret"},
    }
    namespace = handlers.namespace
    namespace["process_multipart"].return_value = {"json_data": payload, "positive": b"conditioning"}
    request = SimpleNamespace(json=AsyncMock(return_value=payload), multipart=AsyncMock(), headers={})

    def finish(item):
        ConnectCache.put_subitem(item[1], "finished", True)
        ConnectCache.put_subitem(item[1], "image", b"image")

    handlers.server.prompt_queue.put.side_effect = finish
    response = asyncio.run(namespace[route](request))

    assert response.status == 200
    handlers.execution.validate_prompt.assert_awaited_once_with("local-task-123", prompt, ["output"])
    handlers.server.node_replace_manager.apply_replacements.assert_called_once_with(prompt)
    item = handlers.server.prompt_queue.put.call_args.args[0]
    assert len(item) == 6
    assert item[1] == "local-task-123"
    assert item[4] == ["output"]
    assert item[5] == {"auth_token_comfy_org": "secret"}
    assert "auth_token_comfy_org" not in item[3]
    assert isinstance(item[3]["create_time"], int)
    if route == "excute_sampling_task":
        assert ConnectCache.get_item("local-task-123") == {}


def test_sampling_validation_failure_does_not_queue(handlers):
    namespace = handlers.namespace
    namespace["process_multipart"].return_value = {"json_data": {"prompt": {}}}
    handlers.execution.validate_prompt.return_value = (False, {"type": "invalid_prompt"}, [], {})
    response = asyncio.run(namespace["excute_sampling_task"](SimpleNamespace(multipart=AsyncMock())))
    assert response.status == 400
    handlers.server.prompt_queue.put.assert_not_called()


def test_history_response_preserves_stored_prompt(handlers):
    history = {"local-task-123": {"prompt": ["private workflow"], "outputs": {}}}
    response = handlers.namespace["_get_history_response"](history)
    assert json.loads(response.body)["local-task-123"]["prompt"] is None
    assert history["local-task-123"]["prompt"] == ["private workflow"]
