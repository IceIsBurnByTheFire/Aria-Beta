<#
    First-run setup. Written for someone who has never seen this project.

    The guiding rule is that every failure has to name the thing to install and where to
    get it. A setup script that says "npm: command not found" and stops has moved the
    problem rather than solved it, and the person it stopped is the person least able to
    work out what happened.

    Nothing here is destructive. It installs into this folder, downloads models into
    core/models and overlay/assets, and writes core/.env only if it does not already
    exist — so running it twice is safe, and running it after a failed attempt picks up
    where it stopped.

    Usage:
        powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
        powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Check
#>

param(
    # Diagnose only. Changes nothing — for when it worked yesterday and doesn't today.
    [switch] $Check
)

$ErrorActionPreference = 'Stop'
$root    = Split-Path -Parent $PSScriptRoot
$core    = Join-Path $root 'core'
$overlay = Join-Path $root 'overlay'

$script:problems = @()
$script:notes    = 0

function Say($text)     { Write-Host $text }
function Step($text)    { Write-Host ""; Write-Host $text -ForegroundColor Cyan }
function Good($text)    { Write-Host "  [ok]   $text" -ForegroundColor Green }
function Warn($text)    {
    Write-Host "  [note] $text" -ForegroundColor Yellow
    $script:notes++
}
function Bad($what, $fix) {
    Write-Host "  [need] $what" -ForegroundColor Red
    Write-Host "         $fix" -ForegroundColor Yellow
    $script:problems += $what
}

function Have($name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }

function Confirm($question) {
    # Default yes. Everything asked here is something the person double-clicked this
    # script in order to get, so Enter should be the answer that continues.
    $answer = Read-Host "$question [Y/n]"
    return ($answer.Trim() -eq '' -or $answer.Trim() -match '^[Yy]')
}

function Reset-PathFromMachine {
    <#
        Pick up a PATH that an installer changed while this script was running.

        A fresh `winget install` writes the new entry to the registry, and this process
        started before that happened. Without re-reading it, the `Have 'ollama'` check
        immediately below fails on a *successful* install, and the script tells you to
        install the thing it just installed.
    #>
    $env:Path = ([Environment]::GetEnvironmentVariable('Path', 'Machine'), `
                 [Environment]::GetEnvironmentVariable('Path', 'User')) -join ';'
}

function Test-LocalPort($port) {
    # A refused connection on loopback returns immediately, so this costs nothing when
    # the answer is no. Same helper as start-aria.ps1 - two small copies rather than a
    # third file that both have to find.
    $client = New-Object System.Net.Sockets.TcpClient
    try { $client.Connect('127.0.0.1', $port); return $true }
    catch { return $false }
    finally { $client.Dispose() }
}

function Get-DefaultModel($corePath) {
    <#
        Which model core will actually ask Ollama for.

        Read from core rather than written down here. The previous version of this
        script hardcoded a name, core's default drifted away from it, and the result was
        a setup that completed cleanly and an Aria that could not answer - the two files
        disagreed and nothing compared them.
    #>
    try {
        $name = uv run --directory $corePath python -c `
            "from aria.config import LLMConfig; print(LLMConfig().model)" 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        return ($name | Select-Object -Last 1).Trim()
    } catch { return $null }
}

function Test-OllamaModel($name) {
    <#
        Is this exact model already downloaded?

        Answered from the manifest directory rather than by running `ollama list`,
        because `ollama list` starts the server when it is not already up. In -Check
        that would be a diagnostic changing the thing it is diagnosing - it would start
        Ollama and then cheerfully report Ollama as running - and -Check states in its
        own help that it changes nothing.

        Exact tag, not a substring: `qwen3:8b` and `qwen3:14b` share a name, and pulling
        the wrong one is a 5 GB mistake made silently.
    #>
    $root = if ($env:OLLAMA_MODELS) { $env:OLLAMA_MODELS }
            else { Join-Path $env:USERPROFILE '.ollama\models' }

    $repo, $tag = $name -split ':', 2
    if (-not $tag) { $tag = 'latest' }
    # An unqualified name lives under `library`; `someone/thing` under `someone`.
    if ($repo -notmatch '/') { $repo = "library/$repo" }

    return Test-Path (Join-Path $root "manifests\registry.ollama.ai\$repo\$tag")
}

# --- what has to be on the machine already -----------------------------------
Step "Checking what's installed"

if (Have 'uv') {
    Good "uv  $((uv --version) -replace '^uv ', '')"
} else {
    Bad "uv is not installed (it manages Python for this project)" `
        "Install it: winget install --id=astral-sh.uv  -- then reopen this window."
}

if (Have 'node') {
    $nodeMajor = ((node --version) -replace '^v', '').Split('.')[0] -as [int]
    if ($nodeMajor -ge 18) { Good "Node $(node --version)" }
    else { Bad "Node $(node --version) is too old (18+ needed)" `
               "Update it: winget install --id=OpenJS.NodeJS.LTS" }
} else {
    Bad "Node.js is not installed (it runs the character window)" `
        "Install it: winget install --id=OpenJS.NodeJS.LTS  -- then reopen this window."
}

# --- what the machine can actually run ---------------------------------------
Step "Checking the hardware"

$gpu = $null
if (Have 'nvidia-smi') {
    $gpu = (nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null |
            Select-Object -First 1)
}
if ($gpu) {
    $parts = $gpu.Split(',')
    $vram  = [int]($parts[1].Trim())
    Good "$($parts[0].Trim()) - $([math]::Round($vram/1024)) GB"
    if ($vram -ge 6000) {
        Say  "         Enough for speech recognition and a local chat model."
    } else {
        Warn "Under 6 GB, so a local chat model will be a squeeze."
        Say  "         Speech still runs on the GPU. See 'Chat model' in README.md."
    }
} else {
    Warn "No NVIDIA GPU found."
    Say  "         Speech recognition will run on the processor - slower, but it works."
    Say  "         A local chat model will not: expect minutes per reply, not seconds."
    Say  "         Use a free cloud model instead. See 'Chat model' in README.md."
}

if ($script:problems.Count -gt 0) {
    Write-Host ""
    Write-Host "  Install the things marked [need] above, then run this again." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

if ($Check) {
    Step "Checking what's already set up"
    $models = @('silero_vad.onnx', 'kokoro-v1.0.onnx', 'voices-v1.0.bin')
    foreach ($m in $models) {
        if (Test-Path (Join-Path $core "models\$m")) { Good "core\models\$m" }
        else { Bad "core\models\$m is missing" "Run setup.ps1 without -Check." }
    }
    if (Test-Path (Join-Path $core '.venv')) { Good "core\.venv" }
    else { Bad "core\.venv is missing" "Run setup.ps1 without -Check." }
    if (Test-Path (Join-Path $overlay 'node_modules')) { Good "overlay\node_modules" }
    else { Bad "overlay\node_modules is missing" "Run setup.ps1 without -Check." }

    # Separately from node_modules, because the two fail independently and only this one
    # fails invisibly. See the note beside the same check in the install section.
    Push-Location $overlay
    try {
        node -e "require('electron')" 2>$null
        if ($LASTEXITCODE -eq 0) { Good "Electron binary" }
        else {
            Bad "Electron's binary is missing (the package is there, the .exe is not)" `
                "Run:  npm install --prefix `"$overlay`" --force"
        }
    } finally { Pop-Location }

    if (Test-Path (Join-Path $overlay 'src\vendor\live2dcubismcore.min.js')) {
        Good "the Live2D runtime"
    } else {
        Bad "the Live2D runtime is missing" `
            "Run setup.ps1 without -Check. Without it the character window is blank."
    }
    if (Test-Path (Join-Path $core '.env')) { Good "core\.env" }
    else { Warn "core\.env is missing - she'll run on defaults, no Discord." }

    # The local chat model, which is the other half of "it worked yesterday". Warnings
    # rather than [need]: a cloud key in core\.env is a perfectly good answer to all of
    # this, and -Check cannot tell which way you went without parsing .env.
    if (Have 'ollama') {
        Good "ollama"
        $model = Get-DefaultModel $core
        if (-not $model) {
            Warn "Could not ask core which model it wants (is core\.venv set up?)."
        } elseif (Test-OllamaModel $model) {
            Good "$model"
        } else {
            Warn "$model is not downloaded."
            Say  "         Run:  ollama pull $model"
        }
        if (Test-LocalPort 11434) { Good "Ollama is running" }
        else { Warn "Ollama is not running. Start Aria.bat starts it for you." }
    } else {
        Warn "Ollama is not installed - she has no local model to think with."
        Say  "         Run setup.ps1 without -Check, or use a cloud key (README.md)."
    }
    Write-Host ""
    if ($script:problems.Count -eq 0) {
        # "All present" printed directly under "the model is not downloaded" is a
        # summary contradicting the line above it. A note is not a broken install - a
        # cloud key answers most of them - but it is also not nothing, and the summary
        # should not talk over it.
        if ($script:notes -eq 0) { Write-Host "  All present." -ForegroundColor Green }
        else { Write-Host "  Nothing broken - but see the notes above." -ForegroundColor Yellow }
    }
    Write-Host ""
    exit ([int]($script:problems.Count -gt 0))
}

# --- install -----------------------------------------------------------------
Step "Setting up the voice loop (this downloads a few hundred MB)"
Push-Location $core
try {
    uv sync
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }
    Good "Python packages installed"

    # Kokoro and Silero do not fetch themselves the way Whisper does.
    uv run python -m aria.setup_models
    if ($LASTEXITCODE -ne 0) { throw "model download failed" }
    Good "speech models downloaded"
} finally { Pop-Location }

Step "Setting up the character window"
Push-Location $overlay
try {
    npm install
    if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
    Good "Node packages installed"

    # `npm install` reporting success is not evidence that Electron is usable.
    #
    # Electron ships a stub package and downloads a 110 MB binary from a postinstall
    # script. When that script fails, npm still exits 0 and node_modules\electron still
    # exists - it just has no electron.exe in it. Then the launcher starts the overlay
    # with -WindowStyle Hidden, the process dies in under a second, and what you get is
    # a working voice loop and no character, with no error anywhere on screen.
    #
    # Seen for real, and not a network problem: electron 33 pins extract-zip 2.0.1,
    # whose extraction step silently does nothing on Node 24. The download succeeded,
    # the unpack was a no-op, and every visible signal said the install had worked.
    node -e "require('electron')" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Electron installed but its binary is missing - the character window cannot start.`n" +
              "         Try:  npm install --prefix `"$overlay`" --force"
    }
    Good "Electron ready"

    # The Live2D runtime is not redistributable and the art is not ours, so both come
    # down at setup time. Skipping this leaves a window that renders nothing at all,
    # with no error - which is the single most confusing way this can fail.
    #
    # Said before the download rather than after it, because "you are about to accept
    # someone's terms" is only a real notice while there is still a chance to stop.
    Say ""
    Say "  The character and the Live2D runtime are Live2D's, not this project's."
    Say "  Downloading them means accepting Live2D's terms: free for individuals and"
    Say "  for businesses under 10,000,000 JPY of annual revenue, no redistributing"
    Say "  the character, and limits on what she may be shown saying."
    Say "    https://www.live2d.com/eula/live2d-free-material-license-agreement_en.html"
    Say "    https://www.live2d.com/eula/live2d-sample-model-terms_en.html"
    Say "  Short version: THIRD-PARTY-NOTICES.md"
    Say ""

    npm run fetch-assets
    if ($LASTEXITCODE -ne 0) { throw "asset download failed" }
    Good "Live2D runtime and the Haru character downloaded"
} finally { Pop-Location }

# --- settings ----------------------------------------------------------------
Step "Settings"
$envPath = Join-Path $core '.env'
if (Test-Path $envPath) {
    Good "core\.env already exists - left alone"
} else {
    Copy-Item (Join-Path $core '.env.example') $envPath
    Good "created core\.env from the example"
    Say  "         Everything in it is optional. Open it if you want Discord or a"
    Say  "         cloud model; ignore it otherwise."
}

# --- what's left, and it depends on the machine ------------------------------
Step "One thing left: where her words come from"
if ($gpu -and [int]($gpu.Split(',')[1].Trim()) -ge 6000) {
    Say "  Your GPU can run a model locally, which is private and free."
    Say ""

    # Asked for and pulled here rather than printed as two commands to run later.
    # Printing them is what this script used to do, and it named `llama3.1:8b` while
    # core defaulted to something else entirely - so following the instructions exactly
    # produced "that model is not installed" on the first thing you said. Anything with
    # a model name in it has to come from core, which is the only thing that knows.
    if (-not (Have 'ollama')) {
        Say "  Ollama runs the model. It is not installed."
        if (Confirm "  Install it now with winget?") {
            winget install --id=Ollama.Ollama --accept-source-agreements --accept-package-agreements
            # winget returns non-zero for "already installed" among other things, so the
            # question is whether the command exists now, not what winget thought.
            Reset-PathFromMachine
            if (Have 'ollama') { Good "Ollama installed" }
            else {
                Warn "Ollama still is not on PATH."
                Say  "         It may need a new terminal, or a manual install from"
                Say  "         https://ollama.com/download - then run this again."
            }
        } else {
            Say "  Skipped. Install it later with:  winget install --id=Ollama.Ollama"
        }
    } else {
        Good "Ollama is installed"
    }

    if (Have 'ollama') {
        $model = Get-DefaultModel $core
        if (-not $model) {
            Warn "Could not ask core which model it wants - skipping the download."
            Say  "         Start her once and she will tell you the exact pull command."
        } elseif (Test-OllamaModel $model) {
            Good "$model is already downloaded"
        } else {
            Say ""
            Say "  She needs a model to talk with: $model"
            Say "  That is roughly 5 GB, downloaded once."
            if (Confirm "  Download it now?") {
                ollama pull $model
                if ($LASTEXITCODE -eq 0) { Good "$model downloaded" }
                else { Warn "The download did not finish. Re-run:  ollama pull $model" }
            } else {
                Say "  Skipped. Download it later with:  ollama pull $model"
            }
        }
    }

    Say ""
    Say "  Or skip all of it and use a free cloud key instead - see 'Chat model' in README.md."
} else {
    Say "  This machine needs a cloud model for the chat part. It's free and takes"
    Say "  about two minutes:"
    Say ""
    Say "    1. Get a key at https://console.groq.com/keys"
    Say "    2. Put it in core\.env as   GROQ_API_KEY=..."
    Say "    3. Set                      ARIA_LLM_BACKEND=groq"
    Say ""
    Say "  Your voice and her voice still run on your PC. Only the text of the"
    Say "  conversation is sent, and only to the provider you pick."
}

Write-Host ""
Write-Host "  Setup done. Start her with:  Start Aria.bat" -ForegroundColor Green
Write-Host ""
