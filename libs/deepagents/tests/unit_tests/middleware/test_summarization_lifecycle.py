"""Public graph behavior when concurrent history offload outlives a summary."""

import asyncio
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import pytest
from langchain_core.callbacks import AsyncCallbackHandler, AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langgraph.errors import GraphBubbleUp, GraphDrained
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import override

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import EditResult, FileDownloadResponse, FileUploadResponse, WriteResult
from deepagents.middleware.summarization import SummarizationMiddleware, SummarizationToolMiddleware


class _Model(FakeMessagesListChatModel):
    failure: Exception | None = None

    def bind_tools(self, tools: Sequence[object], **kwargs: object) -> "_Model":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        if self.failure is not None:
            raise self.failure
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        if self.failure is not None:
            raise self.failure
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


class _HeldHistory(FilesystemBackend):
    def __init__(self, root: Path, failure: Exception | None = None) -> None:
        super().__init__(root_dir=root, virtual_mode=True)
        self.failure = failure
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.cancelled = False

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        self.started.set()
        try:
            await self.release.wait()
            if self.failure is not None:
                raise self.failure
            return await super().awrite(file_path, content)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.finished.set()


class _FailedHistory(FilesystemBackend):
    def __init__(self, root: Path, operation: Literal["download", "write", "edit", "upload"], failure: Exception) -> None:
        super().__init__(root_dir=root, virtual_mode=True)
        self.operation = operation
        self.failure = failure

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        if self.operation == "download":
            raise self.failure
        if self.operation == "edit":
            return [FileDownloadResponse(path=path, content=b"Earlier history\n") for path in paths]
        return super().download_files(paths)

    def write(self, file_path: str, content: str) -> WriteResult:
        if self.operation == "write":
            raise self.failure
        return super().write(file_path, content)

    @override
    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        if self.operation == "edit":
            raise self.failure
        return super().edit(file_path, old_string, new_string, replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        if self.operation == "upload":
            raise self.failure
        return super().upload_files(files)


class _ReleaseAfterFailure(AsyncCallbackHandler):
    run_inline = True
    raise_error = True

    def __init__(self, failure: Exception, backend: _HeldHistory) -> None:
        self.failure = failure
        self.backend = backend
        self.release: asyncio.TimerHandle | None = None

    async def on_chain_error(self, error: BaseException, **kwargs: object) -> None:
        if error is self.failure and self.release is None:
            self.release = asyncio.get_running_loop().call_later(0.1, self.backend.release.set)


def _graph(backend: FilesystemBackend, failure: Exception | None = None, *, manual: bool = False) -> CompiledStateGraph:
    summary = SummarizationMiddleware(
        model=_Model(responses=[AIMessage(content="Summary")], failure=failure),
        backend=backend,
        trigger=("messages", 6 if manual else 2),
        keep=("messages", 1),
    )
    middleware = [summary]
    responses = [AIMessage(content="Complete")]
    if manual:
        middleware.append(SummarizationToolMiddleware(summary))
        responses.insert(0, AIMessage(content="", tool_calls=[{"name": "compact_conversation", "args": {}, "id": "compact-1"}]))
    return create_deep_agent(model=_Model(responses=responses), backend=backend, middleware=middleware)


def _messages() -> list[BaseMessage]:
    return [HumanMessage(content="First question"), AIMessage(content="First answer"), HumanMessage(content="Next question")]


def _media_messages() -> list[BaseMessage]:
    image = {
        "type": "image",
        "base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "mime_type": "image/png",
    }
    return [HumanMessage(content=[image]), *_messages()[1:]]


@pytest.mark.parametrize("failure", [RuntimeError("summary failed"), GraphDrained("stop")], ids=["summary-error", "graph-drain"])
async def test_summary_failure_waits_for_admitted_history(tmp_path: Path, failure: Exception) -> None:
    backend = _HeldHistory(tmp_path)
    graph = _graph(backend, failure)
    callback = _ReleaseAfterFailure(failure, backend)
    try:
        with pytest.raises(type(failure)) as raised:
            await graph.ainvoke({"messages": _messages()}, config={"callbacks": [callback]})
        assert raised.value is failure
        assert backend.started.is_set()
        assert backend.finished.is_set()
        assert not backend.cancelled
        assert (await backend.aglob("/conversation_history/*.md")).matches
    finally:
        backend.release.set()
        await asyncio.wait_for(backend.finished.wait(), timeout=2)
        if callback.release is not None:
            callback.release.cancel()


async def test_successful_summary_retains_completed_history(tmp_path: Path) -> None:
    backend = _HeldHistory(tmp_path)
    backend.release.set()

    result = await _graph(backend).ainvoke({"messages": _messages()})

    assert result["messages"][-1].text == "Complete"
    assert backend.finished.is_set()
    assert not backend.cancelled
    history = await backend.aglob("/conversation_history/*.md")
    assert history.matches is not None
    assert len(history.matches) == 1
    content = (await backend.adownload_files([history.matches[0]["path"]]))[0].content
    assert content is not None
    assert b"First question" in content


async def test_summary_continues_after_history_write_failure(tmp_path: Path) -> None:
    backend = _HeldHistory(tmp_path, OSError("history storage unavailable"))
    backend.release.set()

    with pytest.warns(UserWarning, match="Offloading conversation history to backend failed"):
        result = await _graph(backend).ainvoke({"messages": _messages()})

    assert result["messages"][-1].text == "Complete"
    assert backend.finished.is_set()
    assert not backend.cancelled
    assert not (await backend.aglob("/conversation_history/*.md")).matches


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("failure", [GraphDrained("stop"), GraphBubbleUp("control")], ids=["drain", "bubble-up"])
async def test_manual_compaction_preserves_graph_control(tmp_path: Path, failure: Exception, *, asynchronous: bool) -> None:
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    graph = _graph(backend, failure, manual=True)

    with pytest.raises(type(failure)) as raised:
        await graph.ainvoke({"messages": _messages()}) if asynchronous else graph.invoke({"messages": _messages()})

    assert raised.value is failure


@pytest.mark.parametrize("manual", [False, True], ids=["automatic", "manual"])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("operation", ["download", "write", "edit"])
@pytest.mark.parametrize("failure", [GraphDrained("stop"), GraphBubbleUp("control")], ids=["drain", "bubble-up"])
async def test_history_io_preserves_graph_control(
    tmp_path: Path, operation: Literal["download", "write", "edit"], failure: Exception, *, manual: bool, asynchronous: bool
) -> None:
    graph = _graph(_FailedHistory(tmp_path, operation, failure), manual=manual)

    with pytest.raises(type(failure)) as raised:
        await graph.ainvoke({"messages": _messages()}) if asynchronous else graph.invoke({"messages": _messages()})

    assert raised.value is failure


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("failure", [GraphDrained("stop"), GraphBubbleUp("control")], ids=["drain", "bubble-up"])
async def test_media_upload_preserves_graph_control(tmp_path: Path, failure: Exception, *, asynchronous: bool) -> None:
    graph = _graph(_FailedHistory(tmp_path, "upload", failure))

    with pytest.raises(type(failure)) as raised:
        await graph.ainvoke({"messages": _media_messages()}) if asynchronous else graph.invoke({"messages": _media_messages()})

    assert raised.value is failure


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_manual_compaction_reports_ordinary_summary_errors(tmp_path: Path, *, asynchronous: bool) -> None:
    graph = _graph(FilesystemBackend(root_dir=tmp_path, virtual_mode=True), RuntimeError("summary unavailable"), manual=True)

    result = await graph.ainvoke({"messages": _messages()}) if asynchronous else graph.invoke({"messages": _messages()})

    assert result["messages"][-1].text == "Complete"
    tool = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    assert "Compaction failed" in tool.text
    assert "summary unavailable" in tool.text


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_manual_compaction_keeps_small_conversations(tmp_path: Path, *, asynchronous: bool) -> None:
    graph = _graph(FilesystemBackend(root_dir=tmp_path, virtual_mode=True), AssertionError("summary must not run"), manual=True)
    supplied = {"messages": [HumanMessage(content="A short question")]}

    result = await graph.ainvoke(supplied) if asynchronous else graph.invoke(supplied)

    assert result["messages"][-1].text == "Complete"
    tool = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    assert "Nothing to compact yet" in tool.text


@pytest.mark.parametrize("manual", [False, True], ids=["automatic", "manual"])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("operation", ["download", "write", "edit"])
async def test_history_io_remains_best_effort(
    tmp_path: Path, operation: Literal["download", "write", "edit"], *, manual: bool, asynchronous: bool
) -> None:
    backend = _FailedHistory(tmp_path, operation, OSError("history unavailable"))
    graph = _graph(backend, manual=manual)
    warning = (
        pytest.warns(UserWarning, match="Offloading conversation history to backend failed")
        if not manual and operation != "download"
        else nullcontext()
    )

    with warning:
        result = await graph.ainvoke({"messages": _messages()}) if asynchronous else graph.invoke({"messages": _messages()})

    assert result["messages"][-1].text == "Complete"


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_media_upload_failure_retains_placeholder(tmp_path: Path, *, asynchronous: bool) -> None:
    backend = _FailedHistory(tmp_path, "upload", OSError("media unavailable"))
    graph = _graph(backend)

    with pytest.warns(UserWarning, match="media block\\(s\\) could not be offloaded"):
        result = await graph.ainvoke({"messages": _media_messages()}) if asynchronous else graph.invoke({"messages": _media_messages()})

    assert result["messages"][-1].text == "Complete"
    history = backend.glob("/conversation_history/*.md").matches
    assert history is not None
    content = backend.download_files([history[0]["path"]])[0].content
    assert content is not None
    assert b"failed_to_offload" in content
