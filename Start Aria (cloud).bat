@echo off
rem Start Aria with the conversation model running in the cloud.
rem
rem "Start Aria.bat" is the local one and stays the default. This is the same Aria in
rem every other way - same voice, same memory, same character, same Discord bot. Only
rem where the words are generated changes.
rem
rem WHICH provider is set by ARIA_LLM_BACKEND in core\.env, so you can switch without
rem touching this file. Nothing is passed here on purpose:
rem
rem   ARIA_LLM_BACKEND=groq         30/min, 1000/day. Fastest, most generous.  <- default
rem   ARIA_LLM_BACKEND=google       best models; free tier trains on your content.
rem   ARIA_LLM_BACKEND=openrouter   widest model choice, only 50/day.
rem
rem Each needs its own key in core\.env - GROQ_API_KEY, GEMINI_API_KEY or
rem OPENROUTER_API_KEY. She says which one is missing and where to get it.
rem
rem See every provider's free tier and live model list with:
rem   uv run --directory core python -m aria --list-cloud-models
rem
rem Worth knowing before you leave it running:
rem   - Replies are slower to start than local. Fine on Discord, felt out loud.
rem   - Everything she is told goes to a third party, including her notes about you.
rem   - When the daily limit runs out she says so; close her and open "Start Aria.bat".
rem
rem Arguments pass straight through:  "Start Aria (cloud).bat" --speaker-mode

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-aria.ps1" %*
