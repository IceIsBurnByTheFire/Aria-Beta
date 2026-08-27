@echo off
rem Double-click to start Aria: voice loop and character together.
rem
rem This is the LOCAL one - the model runs on your GPU. Private, offline, no limits,
rem and what the latency budget was tuned against. It is also what starts at login.
rem "Start Aria (cloud).bat" runs the same Aria against a much larger model over
rem OpenRouter instead; see the notes in that file for what you give up.
rem
rem This exists only so the PowerShell script is double-clickable. .ps1 files open in
rem an editor by default and are blocked by the default execution policy, so a plain
rem batch wrapper is the difference between "one click" and "explain PowerShell to
rem the user".
rem
rem Arguments pass straight through:  "Start Aria.bat" --speaker-mode --wake-word

rem --llm-backend ollama is passed explicitly so this stays the local one even when
rem ARIA_LLM_BACKEND in core\.env is pointed at a cloud provider. Otherwise setting up
rem cloud mode would silently change what the login shortcut starts.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-aria.ps1" --llm-backend ollama %*
