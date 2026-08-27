@echo off
rem Start Aria with no character on screen - just her voice and the control panel.
rem
rem Everything else is unchanged: she listens, answers, remembers, and is on Discord
rem exactly as usual. The only thing missing is the Live2D window sitting over your
rem work. Useful when you want her listening but need the screen.
rem
rem The control panel is the whole interface here:
rem   Ctrl+Shift+A, or the tray icon (a small dot near the clock).
rem Closing the panel does NOT stop her - reopen it from the tray. Close THIS window,
rem or Ctrl+C in it, to stop her properly.
rem
rem Uses the local model, like "Start Aria.bat". For the cloud one add --llm-backend
rem with the provider from core\.env, or edit the line below.
rem
rem Arguments pass straight through:  "Start Aria (voice only).bat" --speaker-mode

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-aria.ps1" -PanelOnly --llm-backend ollama %*
