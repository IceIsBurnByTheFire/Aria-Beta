@echo off
rem Start Aria with no character on screen AND the cloud model - voice plus the control
rem panel, nothing over your work.
rem
rem This is the two options combined:
rem   no character   like "Start Aria (voice only).bat"
rem   cloud model    like "Start Aria (cloud).bat"
rem
rem WHICH provider comes from ARIA_LLM_BACKEND in core\.env, so switching needs no edit
rem here. Nothing about the backend is passed on the line below, on purpose.
rem
rem The control panel is the whole interface here:
rem   Ctrl+Shift+A, or the tray icon (a small dot near the clock).
rem Closing the panel does NOT stop her - reopen it from the tray. Close THIS window,
rem or Ctrl+C in it, to stop her properly.
rem
rem Arguments pass straight through:  "Start Aria (voice only, cloud).bat" --speaker-mode

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-aria.ps1" -PanelOnly %*
