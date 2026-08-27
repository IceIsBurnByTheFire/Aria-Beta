<#
    Start Aria: the voice loop and the character, together.

    Core runs in the foreground so this window is her transcript — you can watch the
    turns and the latency line as she talks. The overlay is launched behind it and
    shut down when this window closes, because an orphaned character still floating
    on the desktop after you quit is the worst of both worlds.

    Killing the overlay is done by PID tree, never by image name. `npm start` spawns
    Electron as a grandchild, so stopping npm alone leaves her on screen — but
    `taskkill /IM electron.exe` would take VS Code, Discord and Slack down with her.
    /T on the actual PID is the only version of this that is safe.

    Shutdown is belt and braces, because the `finally` below is not guaranteed to run.
    Ctrl+C reaches it; closing the window with the X does not — Windows terminates
    PowerShell and the block is skipped. That leaves an orphaned overlay, which is the
    worst possible failure here: she is still on screen, still blinking, and nothing
    behind her. No mic, no model, no Discord. It reads as "Aria is broken" rather than
    "Aria is not running", and the two have completely different fixes.

    So: a job object kills the overlay when this process dies *however* it dies, and the
    startup sweep clears anything a previous run left behind. Either one alone would have
    prevented the orphan seen on 2026-08-05; both, because the job object is the fix and
    the sweep is what rescues a machine that already has one.
#>

param(
    # Start the control panel without the character window. The voice loop is unchanged;
    # only the Live2D window is skipped.
    [switch] $PanelOnly,

    # Everything after -- goes to the voice loop: --speaker-mode, --wake-word, etc.
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $AriaArgs
)

$ErrorActionPreference = 'Stop'
$root    = Split-Path -Parent $PSScriptRoot
$core    = Join-Path $root 'core'
$overlay = Join-Path $root 'overlay'

function Fail($message, $fix) {
    Write-Host ""
    Write-Host "  $message" -ForegroundColor Red
    Write-Host "  $fix" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to close"
    exit 1
}

function Test-AriaAlreadyRunning($corePath) {
    <#
        Is a voice loop already attached to this checkout?

        Matters much more since she started launching at login: the boot copy is already
        up when you double-click the shortcut out of habit, and a second core cannot bind
        port 8765. That surfaces as an OSError out of `OverlayServer.start()` during
        setup, which reads as Aria crashing on startup rather than as Aria already being
        open. Two processes would also both hold the microphone.

        Matched on the core path plus `-m aria`, so an unrelated Python on the machine
        is never mistaken for her.
    #>
    $pattern = [regex]::Escape($corePath)
    $existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
                Where-Object { $_.CommandLine -and $_.CommandLine -match $pattern -and $_.CommandLine -match '-m\s+aria' }
    return [bool]$existing
}

function Wait-ForOllama($timeoutSeconds = 40) {
    <#
        Give the model backend a moment to come up. Returns $true if it did.

        Ollama launches from the Startup folder too, and two startup items begin at
        roughly the same instant. Aria loses that race about as often as she wins it,
        and the visible result is her booting with a yellow "Cannot reach Ollama" block
        and staying mute until you say something twice.

        Non-fatal by design. Core already degrades cleanly on a missing backend, so
        after the timeout this gets out of the way and lets it print the message it
        wrote for exactly this situation.
    #>
    if (Test-LocalPort 11434) { return $true }

    # Start it rather than only waiting for it.
    #
    # Waiting alone is right when Ollama is coming up on its own from the Startup
    # folder, which is the case this was written for. It is useless in the other one:
    # Ollama installed, nothing launching it, and forty seconds of "waiting..." followed
    # by a message telling you to go and start the thing that is sitting right there.
    #
    # `ollama serve` and not the desktop app, because this is the process she needs and
    # it inherits the job object below - so it goes away with her instead of being left
    # running by a launcher that never announced starting it.
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Write-Host "  Ollama is not installed, so she has no local model to think with." `
            -ForegroundColor DarkYellow
        Write-Host "  Install it:  winget install --id=Ollama.Ollama" -ForegroundColor DarkYellow
        Write-Host "  Or use a free cloud key instead - see 'Chat model' in README.md." `
            -ForegroundColor DarkYellow
        return $false
    }

    Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden | Out-Null
    Write-Host "  She's on screen. Starting Ollama, the local model backend..." `
        -ForegroundColor DarkGray -NoNewline
    $deadline = (Get-Date).AddSeconds($timeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 500
        if (Test-LocalPort 11434) { Write-Host " up." -ForegroundColor DarkGray; return $true }
    }
    Write-Host ""
    Write-Host "  Ollama did not come up, so she won't be able to answer yet." -ForegroundColor DarkYellow
    Write-Host "  Try 'ollama serve' in another terminal - she'll pick it up on the" -ForegroundColor DarkYellow
    Write-Host "  next thing you say." -ForegroundColor DarkYellow
    return $false
}

function Test-LocalPort($port) {
    # A refused connection on loopback returns immediately, so this costs nothing when
    # the answer is no. Cheaper and less fragile than an HTTP probe, which in 5.1 goes
    # looking for proxy settings first.
    $client = New-Object System.Net.Sockets.TcpClient
    try { $client.Connect('127.0.0.1', $port); return $true }
    catch { return $false }
    finally { $client.Dispose() }
}

function Clear-OrphanedOverlay($overlayPath) {
    <#
        Kill a character left on screen by a previous run that did not shut down cleanly.

        Matched on the command line containing *this project's* overlay path, never on
        the image name: `electron.exe` is also VS Code, Discord, Slack and Teams, and a
        launcher that closes your editor is worse than one that leaves a duplicate
        character behind.

        Without this, starting Aria after an unclean exit gives you two overlays stacked
        on top of each other — and the dead one is indistinguishable from the live one.
    #>
    $pattern = [regex]::Escape($overlayPath)
    $orphans = Get-CimInstance Win32_Process -Filter "Name='electron.exe' OR Name='node.exe'" `
                   -ErrorAction SilentlyContinue |
               Where-Object { $_.CommandLine -and $_.CommandLine -match $pattern }
    if (-not $orphans) { return }

    Write-Host "  Clearing an overlay left running by a previous session." -ForegroundColor DarkYellow
    foreach ($o in $orphans) {
        taskkill /PID $o.ProcessId /T /F 2>&1 | Out-Null
    }
    Start-Sleep -Milliseconds 300
}

function New-KillOnCloseJob {
    <#
        A Windows job object that kills everything in it when this process goes away.

        `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` is enforced by the kernel on the *handle*,
        so it fires however PowerShell dies — Ctrl+C, the window's X, a crash, or Task
        Manager. That is precisely the gap the `finally` block cannot cover.

        Returns $null if any of it fails. This is a safety net over an already-working
        shutdown path, and a launcher that refuses to start because it could not create
        a job object would be a worse bug than the one being fixed.
    #>
    try {
        if (-not ('Aria.Win32' -as [type])) {
            Add-Type -Namespace Aria -Name Win32 -MemberDefinition @'
[DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
public static extern IntPtr CreateJobObject(IntPtr a, string name);
[DllImport("kernel32.dll")]
public static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint length);
[DllImport("kernel32.dll")]
public static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
'@
        }
        $job = [Aria.Win32]::CreateJobObject([IntPtr]::Zero, $null)
        if ($job -eq [IntPtr]::Zero) { return $null }

        # JOBOBJECT_EXTENDED_LIMIT_INFORMATION. Only LimitFlags matters, but the whole
        # struct must be allocated at exactly the right size or the call fails — and it
        # fails by returning false, not by throwing, so a wrong size here is a shutdown
        # net that silently is not there. Measured, not guessed:
        #   basic limits 64 + IO_COUNTERS 48 + 4 pointers 32 = 144 on x64
        #   basic limits 48 + IO_COUNTERS 48 + 4 pointers 16 = 112 on x86
        $size  = if ([IntPtr]::Size -eq 8) { 144 } else { 112 }
        $block = [Runtime.InteropServices.Marshal]::AllocHGlobal($size)
        try {
            for ($i = 0; $i -lt $size; $i++) {
                [Runtime.InteropServices.Marshal]::WriteByte($block, $i, 0)
            }
            # LimitFlags sits at offset 16, and 0x2000 is KILL_ON_JOB_CLOSE.
            [Runtime.InteropServices.Marshal]::WriteInt32($block, 16, 0x2000)
            # 9 = JobObjectExtendedLimitInformation
            if (-not [Aria.Win32]::SetInformationJobObject($job, 9, $block, $size)) { return $null }
        }
        finally {
            [Runtime.InteropServices.Marshal]::FreeHGlobal($block)
        }
        return $job
    }
    catch {
        return $null
    }
}

# --- prerequisites, checked before anything is started ----------------------
# Each of these fails in a way that is obvious here and mystifying later: a missing
# node_modules shows up as an Electron crash, missing models as a Python traceback
# thirty lines deep.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Fail "uv is not installed, and the voice loop needs it." `
         "Install from https://docs.astral.sh/uv/ then run this again."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Fail "npm is not installed, and the character window needs it." `
         "Install Node.js from https://nodejs.org/ then run this again."
}
if (-not (Test-Path (Join-Path $overlay 'node_modules'))) {
    Fail "The overlay's dependencies aren't installed." `
         "Run:  npm install --prefix `"$overlay`""
}
# node_modules existing is not the same as Electron working. Electron downloads its
# binary from a postinstall script that can fail while npm still exits 0, leaving a
# package directory with no electron.exe in it. Checked here because the overlay is
# started hidden: without this the process dies in under a second and the only symptom
# is a character that never appears.
if (-not $PanelOnly) {
    Push-Location $overlay
    try { node -e "require('electron')" 2>$null } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) {
        Fail "Electron is installed but its binary is missing, so the character cannot start." `
             "Run:  npm install --prefix `"$overlay`" --force"
    }
}
if (-not (Test-Path (Join-Path $core 'models\kokoro-v1.0.onnx'))) {
    Fail "The speech models aren't downloaded." `
         "Run:  uv run --directory `"$core`" python -m aria.setup_models"
}

# --- go ---------------------------------------------------------------------
if (Test-AriaAlreadyRunning $core) {
    Write-Host ""
    Write-Host "  Aria is already running." -ForegroundColor Yellow
    Write-Host "  Her window is open somewhere - check the taskbar. Close it to stop her." `
        -ForegroundColor DarkGray
    Write-Host ""
    Start-Sleep -Seconds 4
    exit 0
}

Clear-OrphanedOverlay $overlay

# This process joins the job first, so *everything* started below is a member by
# inheritance — the overlay and the voice loop both.
#
# Doing it this way round was found by testing rather than by thinking. Enrolling only
# the overlay fixes the orphan you can see and creates the opposite one: killing the
# launcher leaves core running headless, because terminating a process does not
# terminate its children. Which half survives depends on how the launcher died — the X
# button takes the console down with core and spares the hidden overlay, a hard kill
# does the reverse — and both leave a half-working Aria that looks like a bug in her.
#
# The job kills whatever is left, however it died, and it is enforced by the kernel on
# the handle rather than by any code of ours getting a chance to run.
$job = New-KillOnCloseJob
if ($job) {
    try {
        [void][Aria.Win32]::AssignProcessToJobObject($job, (Get-Process -Id $PID).Handle)
    }
    catch { $job = $null }
}
if (-not $job) {
    # ASCII only inside quotes, deliberately. This file has no BOM, so PowerShell 5.1
    # decodes it as cp1252: an em-dash becomes three characters ending in 0x94, which
    # is a smart closing quote, which PowerShell accepts as a real string delimiter.
    # The string ends early, the rest of the line parses as code, and the whole script
    # fails with "missing closing '}'" twenty lines further down. Safe in comments,
    # never inside a string.
    Write-Host "  (Shutdown safety net unavailable - close with Ctrl+C, not the X.)" `
        -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "  Aria" -ForegroundColor Magenta
if ($PanelOnly) {
    # Read by main.js. Start-Process hands the parent's environment to the child, which
    # is why this is a variable rather than an argument - npm swallows anything after
    # `start` unless it is passed through a `--` that then has to survive two shells.
    $env:ARIA_PANEL_ONLY = "1"
    Write-Host "  voice only - no character on screen. Control panel: Ctrl+Shift+A or the tray." -ForegroundColor DarkGray
} else {
    Write-Host "  overlay starting in the background; this window is her transcript." -ForegroundColor DarkGray
}
Write-Host "  Ctrl+C or close this window to stop both." -ForegroundColor DarkGray
Write-Host ""

# Through cmd.exe, not npm.cmd directly. Start-Process launching a .cmd shim exits
# instantly with -4058 (ENOENT) and the overlay silently never appears — core comes up
# fine, so it reads as "the character is broken" rather than "the launcher is".
$overlayProc = Start-Process -FilePath $env:ComSpec `
    -ArgumentList '/c', 'npm', 'start', '--prefix', "`"$overlay`"" `
    -WindowStyle Hidden -PassThru

# Enrol the overlay explicitly as well as by inheritance. Belt and braces again: the job
# is created before anything is started, so the overlay is already a member by descent,
# but an explicit assign costs nothing and does not depend on that reasoning holding.
if ($job) {
    try   { [void][Aria.Win32]::AssignProcessToJobObject($job, $overlayProc.Handle) }
    catch { }
}

# Say so if the overlay did not survive being started.
#
# It runs hidden, so its stdout goes nowhere anyone will look and a crash on startup is
# indistinguishable from a character that is merely slow to appear. Everything below
# this point works fine without it - core does not need the overlay - so she comes up
# listening and answering with no face, and the report is "Aria is broken" for a fault
# that is entirely in the window.
#
# A second is enough: the failures this catches (a missing binary, a syntax error in
# main.js, a port already held) all happen before Electron finishes booting.
Start-Sleep -Milliseconds 1000
if ($overlayProc.HasExited) {
    Write-Host ""
    Write-Host "  The character window stopped as soon as it started." -ForegroundColor DarkYellow
    Write-Host "  She'll still listen and answer - there is just nothing on screen." -ForegroundColor DarkGray
    Write-Host "  To see why:  npm start --prefix `"$overlay`"" -ForegroundColor DarkGray
    Write-Host ""
}
# Only now, with her already on screen. Waiting for the model backend *before* starting
# the overlay was a 40-second stare at an empty desktop with a console open, which reads
# as "the launcher is broken" and gets closed long before it would have worked. The
# overlay is built to run with no core attached — it keeps breathing and reconnects on a
# backoff — so it should be up while this waits, not after.
[void](Wait-ForOllama)

try {
    # The overlay retries its connection on a backoff, so it does not matter that
    # core is not listening yet.
    if ($AriaArgs) {
        uv run --directory $core python -m aria @AriaArgs
    }
    else {
        uv run --directory $core python -m aria
    }
}
finally {
    if ($overlayProc -and -not $overlayProc.HasExited) {
        # /T for the whole tree: npm is the parent, Electron is the one on screen.
        taskkill /PID $overlayProc.Id /T /F 2>&1 | Out-Null
    }
    Write-Host ""
    Write-Host "  Aria stopped." -ForegroundColor DarkGray
}
