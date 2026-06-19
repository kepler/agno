"""Built-in tool factory functions for Team."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agno.team.team import Team

import asyncio
import contextlib
import json
from copy import copy
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
    cast,
)
from uuid import uuid4

from pydantic import BaseModel

from agno.agent import Agent
from agno.db.base import AsyncBaseDb, BaseDb, SessionType
from agno.exceptions import RunCancelledException
from agno.filters import FilterExpr
from agno.knowledge.types import KnowledgeFilter
from agno.media import Audio, File, Image, Video
from agno.memory import MemoryManager
from agno.models.message import Message, MessageReferences
from agno.run import RunContext
from agno.run.agent import (
    RunCancelledEvent as AgentRunCancelledEvent,
)
from agno.run.agent import (
    RunCompletedEvent as AgentRunCompletedEvent,
)
from agno.run.agent import (
    RunErrorEvent as AgentRunErrorEvent,
)
from agno.run.agent import (
    RunOutput,
    RunOutputEvent,
)
from agno.run.cancel import (
    aregister_member_run,
    araise_if_cancelled,
    raise_if_cancelled,
    register_member_drain_task,
    register_member_run,
)
from agno.run.team import (
    RunCancelledEvent as TeamMemberRunCancelledEvent,
)
from agno.run.team import (
    RunCompletedEvent as TeamMemberRunCompletedEvent,
)
from agno.run.team import (
    RunErrorEvent as TeamMemberRunErrorEvent,
)
from agno.run.team import (
    TeamRunOutput,
    TeamRunOutputEvent,
)
from agno.session import TeamSession
from agno.tools.function import Function
from agno.utils.knowledge import get_agentic_or_user_search_filters
from agno.utils.log import (
    log_debug,
    log_info,
    log_warning,
    use_agent_logger,
    use_team_logger,
)
from agno.utils.merge_dict import merge_dictionaries
from agno.utils.response import (
    check_if_run_cancelled,
)
from agno.utils.team import (
    add_interaction_to_team_run_context,
    format_member_agent_task,
)
from agno.utils.timer import Timer

# Terminal events emitted by a member (Agent or sub-Team); forwarded even when draining after cancel.
_MEMBER_TERMINAL_EVENT_TYPES = (
    AgentRunCancelledEvent,
    AgentRunCompletedEvent,
    AgentRunErrorEvent,
    TeamMemberRunCancelledEvent,
    TeamMemberRunCompletedEvent,
    TeamMemberRunErrorEvent,
)


def _cascading_cancel_run(run_id: str) -> bool:
    """Cancel a run and cascade through children if it's a sub-team.

    Lazy import to avoid circular dependency with `agno.team._run`.
    """
    from agno.team._run import cancel_run as _team_cancel_run

    return _team_cancel_run(run_id)


async def _acascading_cancel_run(run_id: str) -> bool:
    """Async variant of :func:`_cascading_cancel_run`."""
    from agno.team._run import acancel_run as _team_acancel_run

    return await _team_acancel_run(run_id)


def _get_update_user_memory_function(team: "Team", user_id: Optional[str] = None, async_mode: bool = False) -> Function:
    def update_user_memory(task: str) -> str:
        """
        Use this function to submit a task to modify the Agent's memory.
        Describe the task in detail and be specific.
        The task can include adding a memory, updating a memory, deleting a memory, or clearing all memories.

        Args:
            task: The task to update the memory. Be specific and describe the task in detail.

        Returns:
            str: A string indicating the status of the update.
        """
        team.memory_manager = cast(MemoryManager, team.memory_manager)
        response = team.memory_manager.update_memory_task(task=task, user_id=user_id)
        return response

    async def aupdate_user_memory(task: str) -> str:
        """
        Use this function to submit a task to modify the Agent's memory.
        Describe the task in detail and be specific.
        The task can include adding a memory, updating a memory, deleting a memory, or clearing all memories.

        Args:
            task: The task to update the memory. Be specific and describe the task in detail.

        Returns:
            str: A string indicating the status of the update.
        """
        team.memory_manager = cast(MemoryManager, team.memory_manager)
        response = await team.memory_manager.aupdate_memory_task(task=task, user_id=user_id)
        return response

    if async_mode:
        update_memory_function = aupdate_user_memory
    else:
        update_memory_function = update_user_memory  # type: ignore

    return Function.from_callable(update_memory_function, name="update_user_memory")


def _get_chat_history_function(team: "Team", session: TeamSession, async_mode: bool = False):
    def get_chat_history(num_chats: Optional[int] = None) -> str:
        """
        Use this function to get the team chat history in reverse chronological order.
        Leave the num_chats parameter blank to get the entire chat history.
        Example:
            - To get the last chat, use num_chats=1
            - To get the last 5 chats, use num_chats=5
            - To get all chats, leave num_chats blank

        Args:
            num_chats: The number of chats to return.
                Each chat contains 2 messages. One from the team and one from the user.
                Default: None

        Returns:
            str: A JSON string containing a list of dictionaries representing the team chat history.
        """
        import json

        all_chats = session.get_messages(team_id=team.id)

        if len(all_chats) == 0:
            return ""

        history: List[Dict[str, Any]] = [chat.to_dict() for chat in all_chats]  # type: ignore

        if num_chats is not None:
            history = history[-num_chats:]

        return json.dumps(history)

    async def aget_chat_history(num_chats: Optional[int] = None) -> str:
        """
        Use this function to get the team chat history in reverse chronological order.
        Leave the num_chats parameter blank to get the entire chat history.
        Example:
            - To get the last chat, use num_chats=1
            - To get the last 5 chats, use num_chats=5
            - To get all chats, leave num_chats blank

        Args:
            num_chats: The number of chats to return.
                Each chat contains 2 messages. One from the team and one from the user.
                Default: None

        Returns:
            str: A JSON string containing a list of dictionaries representing the team chat history.
        """
        import json

        all_chats = session.get_messages(team_id=team.id)

        if len(all_chats) == 0:
            return ""

        history: List[Dict[str, Any]] = [chat.to_dict() for chat in all_chats]  # type: ignore

        if num_chats is not None:
            history = history[-num_chats:]

        return json.dumps(history)

    if async_mode:
        get_chat_history_func = aget_chat_history
    else:
        get_chat_history_func = get_chat_history  # type: ignore
    return Function.from_callable(get_chat_history_func, name="get_chat_history")


def _update_session_state_tool(team: "Team", run_context: RunContext, session_state_updates: dict) -> str:
    """
    Update the shared session state.  Provide any updates as a dictionary of key-value pairs.
    Example:
        "session_state_updates": {"shopping_list": ["milk", "eggs", "bread"]}

    Args:
        session_state_updates (dict): The updates to apply to the shared session state. Should be a dictionary of key-value pairs.
    """
    if run_context.session_state is None:
        run_context.session_state = {}
    session_state = run_context.session_state
    for key, value in session_state_updates.items():
        session_state[key] = value

    return f"Updated session state: {session_state}"


def _search_past_sessions_function(
    team: "Team",
    num_past_sessions_to_search: Optional[int] = None,
    num_past_session_runs_in_search: Optional[int] = None,
    user_id: Optional[str] = None,
    current_session_id: Optional[str] = None,
    async_mode: bool = False,
) -> Function:
    """Factory for search_past_sessions tool for Team."""

    from agno.agent._default_tools import _extract_session_preview
    from agno.team._init import _has_async_db

    _limit = num_past_sessions_to_search if num_past_sessions_to_search is not None else 20
    _num_runs = num_past_session_runs_in_search if num_past_session_runs_in_search is not None else 3

    def search_past_sessions() -> str:
        """List previous chat sessions with short previews.
        Use read_past_session to read the full conversation for a specific session.

        Returns:
            str: JSON list of session previews with session_id, created_at, and runs (user/assistant pairs).
        """
        if team.db is None:
            return json.dumps([])

        team.db = cast(BaseDb, team.db)
        selected_sessions = team.db.get_sessions(
            session_type=SessionType.TEAM,
            limit=_limit,
            user_id=user_id,
            sort_by="created_at",
            sort_order="desc",
        )

        results: list = []
        for session in selected_sessions:
            if not isinstance(session, TeamSession) or not session.runs:
                continue
            if current_session_id and session.session_id == current_session_id:
                continue
            results.append(_extract_session_preview(session, num_runs=_num_runs))

        return json.dumps(results)

    async def asearch_past_sessions() -> str:
        """List previous chat sessions with short previews.
        Use read_past_session to read the full conversation for a specific session.

        Returns:
            str: JSON list of session previews with session_id, created_at, and runs (user/assistant pairs).
        """
        if team.db is None:
            return json.dumps([])

        if _has_async_db(team):
            selected_sessions = await cast(AsyncBaseDb, team.db).get_sessions(  # type: ignore
                session_type=SessionType.TEAM,
                limit=_limit,
                user_id=user_id,
                sort_by="created_at",
                sort_order="desc",
            )
        else:
            selected_sessions = team.db.get_sessions(  # type: ignore
                session_type=SessionType.TEAM,
                limit=_limit,
                user_id=user_id,
                sort_by="created_at",
                sort_order="desc",
            )

        results: list = []
        for session in selected_sessions:
            if not isinstance(session, TeamSession) or not session.runs:
                continue
            if current_session_id and session.session_id == current_session_id:
                continue
            results.append(_extract_session_preview(session, num_runs=_num_runs))

        return json.dumps(results)

    if async_mode and _has_async_db(team):
        return Function.from_callable(asearch_past_sessions, name="search_past_sessions")
    return Function.from_callable(search_past_sessions, name="search_past_sessions")


def _read_past_session_function(
    team: "Team",
    user_id: Optional[str] = None,
    async_mode: bool = False,
) -> Function:
    """Factory for read_past_session tool for Team."""

    from agno.agent._default_tools import _get_message_text
    from agno.team._init import _has_async_db

    def read_past_session(session_id: str, num_runs: Optional[int] = None) -> str:
        """Read the full conversation from a previous session.
        Use search_past_sessions first to find relevant sessions.

        Args:
            session_id: The session ID to read (from search results).
            num_runs: Maximum number of runs to include. Default: all runs.

        Returns:
            str: The conversation formatted as User/Assistant message pairs.
        """
        if team.db is None:
            return "No database configured."

        team.db = cast(BaseDb, team.db)
        session = team.db.get_session(
            session_id=session_id,
            session_type=SessionType.TEAM,
            user_id=user_id,
        )

        if session is None or not isinstance(session, TeamSession) or not session.runs:
            return "Session not found."

        lines: list = []
        lines.append(f"Session: {session.session_id}")
        if session.created_at:
            lines.append(f"Created: {session.created_at}")
        lines.append("")

        runs = session.runs if num_runs is None else session.runs[:num_runs]
        for run in runs:
            for msg in run.messages or []:
                if msg.role not in ("user", "assistant"):
                    continue
                text = _get_message_text(msg)
                if text:
                    role_label = "User" if msg.role == "user" else "Assistant"
                    lines.append(f"{role_label}: {text}")
                    lines.append("")

        return "\n".join(lines) if lines else "No messages found in session."

    async def aread_past_session(session_id: str, num_runs: Optional[int] = None) -> str:
        """Read the full conversation from a previous session.
        Use search_past_sessions first to find relevant sessions.

        Args:
            session_id: The session ID to read (from search results).
            num_runs: Maximum number of runs to include. Default: all runs.

        Returns:
            str: The conversation formatted as User/Assistant message pairs.
        """
        if team.db is None:
            return "No database configured."

        if _has_async_db(team):
            session = await cast(AsyncBaseDb, team.db).get_session(  # type: ignore
                session_id=session_id,
                session_type=SessionType.TEAM,
                user_id=user_id,
            )
        else:
            session = team.db.get_session(  # type: ignore
                session_id=session_id,
                session_type=SessionType.TEAM,
                user_id=user_id,
            )

        if session is None or not isinstance(session, TeamSession) or not session.runs:
            return "Session not found."

        lines: list = []
        lines.append(f"Session: {session.session_id}")
        if session.created_at:
            lines.append(f"Created: {session.created_at}")
        lines.append("")

        runs = session.runs if num_runs is None else session.runs[:num_runs]
        for run in runs:
            for msg in run.messages or []:
                if msg.role not in ("user", "assistant"):
                    continue
                text = _get_message_text(msg)
                if text:
                    role_label = "User" if msg.role == "user" else "Assistant"
                    lines.append(f"{role_label}: {text}")
                    lines.append("")

        return "\n".join(lines) if lines else "No messages found in session."

    if async_mode and _has_async_db(team):
        return Function.from_callable(aread_past_session, name="read_past_session")
    return Function.from_callable(read_past_session, name="read_past_session")


def _get_delegate_task_function(
    team: "Team",
    run_response: TeamRunOutput,
    run_context: RunContext,
    session: TeamSession,
    team_run_context: Dict[str, Any],
    user_id: Optional[str] = None,
    stream: bool = False,
    stream_events: bool = False,
    async_mode: bool = False,
    input: Optional[str] = None,  # Used for determine_input_for_members=False
    images: Optional[List[Image]] = None,
    videos: Optional[List[Video]] = None,
    audio: Optional[List[Audio]] = None,
    files: Optional[List[File]] = None,
    add_history_to_context: Optional[bool] = None,
    add_dependencies_to_context: Optional[bool] = None,
    add_session_state_to_context: Optional[bool] = None,
    debug_mode: Optional[bool] = None,
) -> Function:
    from agno.team._init import _initialize_member
    from agno.team._run import _update_team_media
    from agno.team._tools import (
        _determine_team_member_interactions,
        _find_member_by_id,
        _get_history_for_member_agent,
        _propagate_member_pause,
    )

    if not images:
        images = []
    if not videos:
        videos = []
    if not audio:
        audio = []
    if not files:
        files = []

    def _setup_delegate_task_to_member(member_agent: Union[Agent, "Team"], task: str):
        # 1. Initialize the member agent

        _initialize_member(team, member_agent)

        # If team has send_media_to_model=False, ensure member agent also has it set to False
        # This allows tools to access files while preventing models from receiving them
        if not team.send_media_to_model:
            member_agent.send_media_to_model = False

        # 2. Handle respond_directly nuances
        if team.respond_directly:
            # Since we return the response directly from the member agent, we need to set the output schema from the team down.
            # Get output_schema from run_context
            team_output_schema = run_context.output_schema if run_context else None
            if not member_agent.output_schema and team_output_schema:
                member_agent.output_schema = team_output_schema

            # If the member will produce structured output, we need to parse the response
            if member_agent.output_schema is not None:
                team._member_response_model = member_agent.output_schema

        # 3. Handle enable_agentic_knowledge_filters on the member agent
        if team.enable_agentic_knowledge_filters and not member_agent.enable_agentic_knowledge_filters:
            member_agent.enable_agentic_knowledge_filters = team.enable_agentic_knowledge_filters

        # 4. Determine team context to send
        team_member_interactions_str = _determine_team_member_interactions(
            team, team_run_context, images=images, videos=videos, audio=audio, files=files
        )

        # 5. Get the team history
        team_history_str = None
        if team.add_team_history_to_members and session:
            team_history_str = session.get_team_history_context(num_runs=team.num_team_history_runs)

        # 6. Create the member agent task or use the input directly
        if team.determine_input_for_members is False:
            member_agent_task = input  # type: ignore
        else:
            member_agent_task = task

        if team_history_str or team_member_interactions_str:
            member_agent_task = format_member_agent_task(  # type: ignore
                task_description=member_agent_task or "",
                team_member_interactions_str=team_member_interactions_str,
                team_history_str=team_history_str,
            )

        # 7. Add member-level history for the member if enabled (because we won't load the session for the member, so history won't be loaded automatically)
        history = None
        if hasattr(member_agent, "add_history_to_context") and member_agent.add_history_to_context:
            history = _get_history_for_member_agent(team, session, member_agent)
            if history:
                if isinstance(member_agent_task, str):
                    history.append(Message(role="user", content=member_agent_task))

        return member_agent_task, history

    def _process_delegate_task_to_member(
        member_agent_run_response: Optional[Union[TeamRunOutput, RunOutput]],
        member_agent: Union[Agent, "Team"],
        member_agent_task: Union[str, Message],
        member_session_state_copy: Dict[str, Any],
    ):
        # Add team run id to the member run

        if member_agent_run_response is not None:
            member_agent_run_response.parent_run_id = run_response.run_id  # type: ignore

        # Update the top-level team run_response tool call to have the run_id of the member run
        if run_response.tools is not None and member_agent_run_response is not None:
            for tool in run_response.tools:
                if tool.tool_name and tool.tool_name.lower() == "delegate_task_to_member":
                    tool.child_run_id = member_agent_run_response.run_id  # type: ignore

        # Update the team run context
        member_name = member_agent.name if member_agent.name else member_agent.id if member_agent.id else "Unknown"
        if isinstance(member_agent_task, str):
            normalized_task = member_agent_task
        elif member_agent_task.content:
            normalized_task = str(member_agent_task.content)
        else:
            normalized_task = ""
        add_interaction_to_team_run_context(
            team_run_context=team_run_context,
            member_name=member_name,
            task=normalized_task,
            run_response=member_agent_run_response,  # type: ignore
        )

        # Add the member run to the team run response if enabled
        if run_response and member_agent_run_response:
            run_response.add_member_run(member_agent_run_response)

        # Scrub the member run based on that member's storage flags before storing
        if member_agent_run_response:
            if (
                not member_agent.store_media
                or not member_agent.store_tool_messages
                or not member_agent.store_history_messages
            ):
                from agno.agent._run import scrub_run_output_for_storage

                scrub_run_output_for_storage(member_agent, run_response=member_agent_run_response)  # type: ignore[arg-type]

            # Add the member run to the team session
            session.upsert_run(member_agent_run_response)

        # Update team session state
        merge_dictionaries(run_context.session_state, member_session_state_copy)  # type: ignore

        # Update the team media
        if member_agent_run_response is not None:
            _update_team_media(team, member_agent_run_response)  # type: ignore

    def delegate_task_to_member(member_id: str, task: str) -> Iterator[Union[RunOutputEvent, TeamRunOutputEvent, str]]:
        """Use this function to delegate a task to the selected team member.
        You must provide a clear and concise description of the task the member should achieve AND the expected output.

        Args:
            member_id (str): The ID of the member to delegate the task to. Use only the ID of the member, not the ID of the team followed by the ID of the member.
            task (str): A clear and concise description of the task the member should achieve.
        Returns:
            str: The result of the delegated task.
        """

        # Find the member agent using the helper function
        result = _find_member_by_id(team, member_id, run_context=run_context)
        if result is None:
            yield f"Member with ID {member_id} not found in the team or any subteams. Please choose the correct member from the list of members:\n\n{team.get_members_system_message_content(indent[...]
            return

        _, member_agent = result
        member_agent_task, history = _setup_delegate_task_to_member(member_agent=member_agent, task=task)

        # Make sure for the member agent, we are using the agent logger
        use_agent_logger()

        member_session_state_copy = copy(run_context.session_state)

        member_agent_run_response = None
        try:
            if stream:
                member_run_id = str(uuid4())
                if run_response.run_id is not None:
                    register_member_run(run_response.run_id, member_run_id)
                member_agent_run_response_stream = member_agent.run(
                    input=member_agent_task if not history else history,
                    user_id=user_id,
                    # All members have the same session_id
                    session_id=session.session_id,
                    session_state=member_session_state_copy,  # Send a copy to the agent
                    images=images,
                    videos=videos,
                    audio=audio,
                    files=files,
                    stream=True,
                    stream_events=stream_events or team.stream_member_events,
                    debug_mode=debug_mode,
                    dependencies=run_context.dependencies,
                    add_dependencies_to_context=add_dependencies_to_context,
                    metadata=run_response.requested by truncated due to length