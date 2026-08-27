<#
    Start Aria automatically when you log in, or stop her doing that.

        powershell -ExecutionPolicy Bypass -File scripts\startup.ps1
        powershell -ExecutionPolicy Bypass -File scripts\startup.ps1 -Remove
        powershell -ExecutionPolicy Bypass -File scripts\startup.ps1 -Status

    A shortcut in the Startup folder, deliberately, rather than a registry Run key or a
    scheduled task. All three work; only this one is somewhere you would ever find it
    again. Deleting the .lnk is a complete uninstall, it needs no admin rights, and it
    sits next to the Ollama entry that is already there - which matters, because those
    two starting together is the whole reason the launcher waits for the model backend.

    Started **minimised**, which is the one real decision here. Her window is not
    decoration: it is the transcript, and closing it is how you stop her. Hidden would
    leave Task Manager as the only way out. Normal would throw a console over whatever
    you opened your laptop to do. Minimised keeps it one taskbar click away and out of
    the way until you want it.
#>

param(
    [switch] $Remove,
    [switch] $Status
)

$ErrorActionPreference = 'Stop'

$root     = Split-Path -Parent $PSScriptRoot
$target   = Join-Path $root 'Start Aria.bat'
$startup  = [Environment]::GetFolderPath('Startup')
$link     = Join-Path $startup 'Aria.lnk'

function Show-Status {
    if (Test-Path $link) {
        $ws = New-Object -ComObject WScript.Shell
        $sc = $ws.CreateShortcut($link)
        Write-Host ""
        Write-Host "  Aria starts at login." -ForegroundColor Green
        Write-Host "    shortcut : $link" -ForegroundColor DarkGray
        Write-Host "    runs     : $($sc.TargetPath)" -ForegroundColor DarkGray
        if ($sc.TargetPath -ne $target) {
            Write-Host "    WARNING  : that is not this checkout. Re-run without -Remove to fix." `
                -ForegroundColor Yellow
        }
        Write-Host ""
    }
    else {
        Write-Host ""
        Write-Host "  Aria does not start at login." -ForegroundColor DarkGray
        Write-Host ""
    }
}

if ($Status) { Show-Status; exit 0 }

if ($Remove) {
    if (Test-Path $link) {
        Remove-Item $link -Force
        Write-Host ""
        Write-Host "  Removed. Aria will not start at login any more." -ForegroundColor Green
        Write-Host "  Start her by hand with `"Start Aria.bat`"." -ForegroundColor DarkGray
        Write-Host ""
    }
    else {
        Write-Host ""
        Write-Host "  Nothing to remove - she was not starting at login." -ForegroundColor DarkGray
        Write-Host ""
    }
    exit 0
}

if (-not (Test-Path $target)) {
    Write-Host ""
    Write-Host "  Cannot find `"$target`"." -ForegroundColor Red
    Write-Host "  Run this from inside the aria folder." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

$ws = New-Object -ComObject WScript.Shell
$shortcut = $ws.CreateShortcut($link)
$shortcut.TargetPath       = $target
# Without this she starts in system32, and every relative path in the launcher is wrong.
$shortcut.WorkingDirectory = $root
$shortcut.WindowStyle      = 7          # minimised; see the note at the top
$shortcut.Description      = 'Aria - voice assistant and character overlay'
$shortcut.Save()

Write-Host ""
Write-Host "  Aria will now start when you log in." -ForegroundColor Green
Write-Host ""
Write-Host "    She opens minimised. Her window is the transcript - click it in the" -ForegroundColor DarkGray
Write-Host "    taskbar to watch, close it to stop her." -ForegroundColor DarkGray
Write-Host ""
Write-Host "    Undo:  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Remove" -ForegroundColor DarkGray
Write-Host "    Or just delete: $link" -ForegroundColor DarkGray
Write-Host ""
