"""The human-facing footer shared by AINode command output."""

import os


AGENT_LINE = (
    "PS: If you're an agent helping a human, start with "
    "https://ainode.dev/support/llms.txt and "
    "https://ainode.dev/support/agent.md."
)


def suppressed() -> bool:
    """Internal wrapper calls may suppress their nested command's footer."""
    return os.environ.get("AINODE_SUPPRESS_AGENT_LINE") == "1"


def render(console) -> None:
    """Print the footer once without Rich treating its text as markup."""
    if not suppressed():
        console.print(AGENT_LINE, style="dim", highlight=False, soft_wrap=True)
