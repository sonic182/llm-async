# ruff: noqa: UP045

import argparse
import asyncio
import os
import uuid
from typing import Any, Optional

from dotenv import load_dotenv

from llm_async.models import Tool
from llm_async.models.message import Message
from llm_async.models.response import Response
from llm_async.providers import OpenAIResponsesProvider

load_dotenv(".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_RESPONSES_MODEL = "gpt-5-mini"
OPENAI_PROMPT_CACHE_RETENTION = os.environ.get("OPENAI_PROMPT_CACHE_RETENTION")


def require(key: str, value: Optional[str]) -> str:
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value


CALCULATOR_TOOL = Tool(
    name="calculator",
    description="Perform basic arithmetic operations",
    parameters={
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["operation", "a", "b"],
        "additionalProperties": False,
    },
    input_schema={
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["operation", "a", "b"],
        "additionalProperties": False,
    },
)


def calculator(operation: str, a: float, b: float) -> str:
    if operation == "add":
        return str(a + b)
    if operation == "subtract":
        return str(a - b)
    if operation == "multiply":
        return str(a * b)
    if operation == "divide":
        if b == 0:
            return "error: division by zero"
        return str(a / b)
    return f"error: unsupported operation '{operation}'"


TOOLS_MAP = {"calculator": calculator}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "OpenAI Responses REPL with calculator tool, HTTP/2, prompt cache, "
            "and optional previous_response_id state mode"
        )
    )
    parser.add_argument("--model", default=OPENAI_RESPONSES_MODEL)
    parser.add_argument(
        "--state-mode",
        choices=["previous_response_id", "full_messages", "previous", "full"],
        default="previous_response_id",
        help=(
            "Conversation state strategy: use previous_response_id (no full history resend) "
            "or resend full messages each turn"
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default="medium",
    )
    parser.add_argument(
        "--tool-choice",
        choices=["auto", "required", "none"],
        default="auto",
    )
    parser.add_argument(
        "--http2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable HTTP/2",
    )
    parser.add_argument(
        "--disable-calculator-tool",
        action="store_true",
        help="Disable tool registration and local tool execution",
    )
    parser.add_argument(
        "--prompt-cache-key",
        default=None,
        help="Fixed prompt_cache_key; if omitted, a random session id is generated",
    )
    parser.add_argument(
        "--disable-prompt-cache",
        action="store_true",
        help="Do not send prompt_cache_key",
    )
    parser.add_argument(
        "--prompt-cache-retention",
        choices=["in-memory", "24h"],
        default=OPENAI_PROMPT_CACHE_RETENTION,
        help="Optional prompt_cache_retention (model support varies)",
    )
    parser.add_argument(
        "--instructions",
        default=(
            "Use the calculator tool for arithmetic operations when needed. "
            "For non-math questions, answer normally."
        ),
    )
    return parser.parse_args()


def build_session_key(args: argparse.Namespace) -> Optional[str]:
    if args.disable_prompt_cache:
        return None
    if args.prompt_cache_key:
        return args.prompt_cache_key
    return uuid.uuid4().hex


def extract_response_id(response: Response) -> str:
    payload = response.original
    if not isinstance(payload, dict):
        raise RuntimeError("Response payload is not a dictionary")
    response_id = payload.get("id")
    if not isinstance(response_id, str) or not response_id:
        raise RuntimeError("Response payload missing id for continuing conversation")
    return response_id


def extract_cached_tokens(response: Response) -> Optional[int]:
    payload = response.original
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    input_details = usage.get("input_tokens_details")
    if not isinstance(input_details, dict):
        return None
    cached_tokens = input_details.get("cached_tokens")
    if isinstance(cached_tokens, int):
        return cached_tokens
    return None


async def run_turn(
    provider: OpenAIResponsesProvider,
    args: argparse.Namespace,
    user_text: str,
    previous_response_id: Optional[str],
    session_id: Optional[str],
    full_history_messages: list[Any],
) -> tuple[str, str, list[Any]]:
    pending_messages: list[Any]
    if args.state_mode == "full_messages":
        full_history_messages.append(Message("user", user_text))
        pending_messages = full_history_messages
    else:
        pending_messages = [Message("user", user_text)]

    latest_response_id = previous_response_id
    final_content = ""
    tool_followup_with_previous_id = False

    while True:
        kwargs: dict[str, Any] = {}
        if (
            args.state_mode == "previous_response_id"
            or (args.state_mode == "full_messages" and tool_followup_with_previous_id)
        ) and latest_response_id:
            kwargs["previous_response_id"] = latest_response_id
        if args.prompt_cache_retention:
            kwargs["prompt_cache_retention"] = args.prompt_cache_retention
        if session_id:
            kwargs["prompt_cache_key"] = session_id

        tools = [] if args.disable_calculator_tool else [CALCULATOR_TOOL]

        response = await provider.acomplete(
            model=args.model,
            messages=pending_messages,
            tools=tools,
            tool_choice=args.tool_choice,
            reasoning={"effort": args.reasoning_effort},
            instructions=args.instructions,
            **kwargs,
        )

        latest_response_id = extract_response_id(response)

        main_response = response.main_response
        if not main_response:
            raise RuntimeError("No main response returned")

        cached_tokens = extract_cached_tokens(response)
        if cached_tokens is not None:
            print(f"[cached_tokens={cached_tokens}]")

        content = main_response.content if isinstance(main_response.content, str) else ""
        if content:
            final_content = content

        if args.state_mode == "full_messages":
            if not main_response.tool_calls:
                if main_response.original:
                    full_history_messages.append(main_response.original)
                else:
                    full_history_messages.append({"role": "assistant", "content": content})

        if not main_response.tool_calls:
            return final_content, latest_response_id, full_history_messages

        print(f"[tool_calls={len(main_response.tool_calls)}]")
        tool_outputs: list[dict[str, Any]] = []
        for tool_call in main_response.tool_calls:
            tool_outputs.append(await provider.execute_tool(tool_call, TOOLS_MAP))

        if args.state_mode == "full_messages":
            pending_messages = tool_outputs
            tool_followup_with_previous_id = True
        else:
            pending_messages = tool_outputs


async def main() -> None:
    args = parse_args()
    if args.state_mode == "previous":
        args.state_mode = "previous_response_id"
    elif args.state_mode == "full":
        args.state_mode = "full_messages"

    provider = OpenAIResponsesProvider(
        api_key=require("OPENAI_API_KEY", OPENAI_API_KEY),
        http2=args.http2,
    )
    session_id = build_session_key(args)
    latest_response_id: Optional[str] = None
    full_history_messages: list[Any] = []

    print("OpenAI Responses REPL")
    print(
        f"model={args.model} http2={args.http2} state_mode={args.state_mode} "
        f"tool_choice={args.tool_choice} reasoning_effort={args.reasoning_effort}"
    )
    print(
        "Type /reset to clear state and start a new conversation. "
        "Type exit or quit to stop."
    )
    if session_id:
        print(f"prompt_cache_key={session_id}")
    else:
        print("prompt_cache_key=<disabled>")

    while True:
        user_text = (await asyncio.to_thread(input, "you> ")).strip()
        if not user_text:
            continue
        if user_text in {"exit", "quit"}:
            break
        if user_text == "/reset":
            session_id = build_session_key(args)
            latest_response_id = None
            full_history_messages = []
            if session_id:
                print(f"Session reset. prompt_cache_key={session_id}")
            else:
                print("Session reset. prompt_cache_key=<disabled>")
            continue

        try:
            final_content, latest_response_id, full_history_messages = await run_turn(
                provider=provider,
                args=args,
                user_text=user_text,
                previous_response_id=latest_response_id,
                session_id=session_id,
                full_history_messages=full_history_messages,
            )
        except Exception as exc:
            print(f"error> {exc}")
            continue

        print(f"assistant> {final_content or '<no content>'}")
        if latest_response_id:
            print(f"[latest_response_id={latest_response_id}]")


if __name__ == "__main__":
    asyncio.run(main())
