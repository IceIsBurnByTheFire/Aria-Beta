@echo off
rem Run this once, before the first time you start her.
rem
rem It checks what you have installed, works out what your PC can run, downloads the
rem speech models and the character, and tells you the one thing left to decide.
rem
rem Safe to run again if something went wrong - it picks up where it stopped.
rem
rem   "Setup.bat" -Check    diagnose only, change nothing

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" %*
pause
