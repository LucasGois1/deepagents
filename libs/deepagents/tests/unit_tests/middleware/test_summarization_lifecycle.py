"""Public graph behavior when concurrent history offload outlives a summary."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
from langchain_core.callbacks import AsyncCallbackHandler, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatResult
from langgraph.errors import GraphDrained
from langgraph.graph.state import CompiledStateGraph

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import WriteResult
from deepagents.middleware.summarization import SummarizationMiddleware


class _Model(FakeMessagesListChatModel):
    failure: Exception | None = None

    def bind_tools(self, tools: Sequence[object], **kwargs: object) -> "_Model":
        return self

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


def _graph(backend: FilesystemBackend, failure: Exception | None = None) -> CompiledStateGraph:
    summary = SummarizationMiddleware(
        model=_Model(responses=[AIMessage(content="Summary")], failure=failure),
        backend=backend,
        trigger=("messages", 2),
        keep=("messages", 1),
    )
    return create_deep_agent(model=_Model(responses=[AIMessage(content="Complete")]), backend=backend, middleware=[summary])


def _messages() -> list[BaseMessage]:
    return [HumanMessage(content="First question"), AIMessage(content="First answer"), HumanMessage(content="Next question")]


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
