"""Is the Discord side actually set up? Connect, report, disconnect.

A simulated conversation cannot tell you about your token, your intents, or whether the
bot has been invited anywhere — and those are the three things that are wrong on a first
run. So this spends one real connection on answering them, and says what to do about
each.

Run:  uv run --directory . python tests/check_discord.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import discord  # noqa: E402

from aria.config import Config  # noqa: E402
from aria.discord_bot import INVITE_PERMISSIONS  # noqa: E402

RED, GREEN, YELLOW, DIM, RESET = (
    "\033[31m", "\033[32m", "\033[33m", "\033[2m", "\033[0m",
)


async def main() -> int:
    cfg = Config().discord
    if not cfg.token:
        print(f"\n{YELLOW}No token.{RESET} Put one in core/.env as ARIA_DISCORD_TOKEN — "
              f"see core/.env.example.\n")
        return 1

    intents = discord.Intents.none()
    intents.guilds = True
    intents.guild_messages = True
    intents.dm_messages = True
    intents.message_content = True
    client = discord.Client(intents=intents)
    problems: list[str] = []

    @client.event
    async def on_ready() -> None:
        print(f"\n{GREEN}Connected{RESET} as {client.user}  (id {client.user.id})")

        if client.guilds:
            print(f"\n  In {len(client.guilds)} server(s):")
            for g in client.guilds:
                print(f"    {g.name}  {DIM}({g.id}){RESET}")
        else:
            problems.append(
                "She is not in any server. You cannot open a DM to a bot you share no\n"
                "  server with, so right now she is unreachable. Invite her:\n"
                f"  https://discord.com/oauth2/authorize?client_id={client.user.id}"
                f"&scope=bot&permissions={INVITE_PERMISSIONS}"
            )

        if not cfg.owner_id:
            problems.append(
                "ARIA_DISCORD_OWNER is not set. She will answer anyone who can reach\n"
                "  her, and will refuse every screen request because nothing identifies\n"
                "  the sender as you. Right-click yourself in Discord → Copy User ID."
            )
        else:
            try:
                owner = await client.fetch_user(cfg.owner_id)
                print(f"\n  Owner: {owner}  {DIM}({owner.id}){RESET}")
                print(f"  {DIM}She answers this account and ignores everyone else.{RESET}")
            except discord.NotFound:
                problems.append(
                    f"ARIA_DISCORD_OWNER is {cfg.owner_id}, which is not a Discord\n"
                    "  account. She will therefore ignore every message, including yours."
                )
            except discord.HTTPException as e:
                print(f"\n  {YELLOW}Could not look up the owner id: {e}{RESET}")

        await client.close()

    try:
        await client.start(cfg.token)
    except discord.LoginFailure:
        print(f"\n{RED}Discord refused the token.{RESET} Reset it in the Developer "
              f"Portal → Bot → Reset Token,\nand put the new one in core/.env.\n")
        return 1
    except discord.PrivilegedIntentsRequired:
        print(f"\n{RED}The Message Content intent is off.{RESET} Developer Portal → your "
              f"app → Bot →\nPrivileged Gateway Intents → MESSAGE CONTENT INTENT. Without "
              f"it every message\narrives empty and she has nothing to answer.\n")
        return 1

    if problems:
        print(f"\n{YELLOW}{'─' * 68}{RESET}")
        for p in problems:
            print(f"{YELLOW}• {p}{RESET}")
        print(f"{YELLOW}{'─' * 68}{RESET}\n")
        return 1

    print(f"\n{GREEN}All set.{RESET} Start Aria and send her a DM.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
