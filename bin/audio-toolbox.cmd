@echo off
rem Windows launcher for the audio-toolbox CLI. The implementation lives inside
rem the skill directory so an installed copy of skills/sam-audio stays self-contained.
if defined PYTHON (set "_AT_PYTHON=%PYTHON%") else (set "_AT_PYTHON=python")
"%_AT_PYTHON%" "%~dp0..\skills\sam-audio\scripts\audio_toolbox.py" %*
exit /b %ERRORLEVEL%