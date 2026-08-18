$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$names = Get-ChildItem -LiteralPath $here -Directory | Where-Object { $_.Name -like "starwhisper-*" }
if (-not $names) {
    throw "no starwhisper-* skill directories under $here"
}
$targets = @(
    (Join-Path $env:USERPROFILE ".codex\skills"),
    (Join-Path $env:USERPROFILE ".cursor\skills")
)
foreach ($t in $targets) {
    New-Item -ItemType Directory -Force -Path $t | Out-Null
    foreach ($n in $names) {
        $dest = Join-Path $t $n.Name
        if (Test-Path -LiteralPath $dest) {
            Remove-Item -LiteralPath $dest -Recurse -Force
        }
        Copy-Item -LiteralPath $n.FullName -Destination $dest -Recurse
        Write-Host "installed $($n.Name) -> $dest"
    }
}
Write-Host "Set STARWHISPER_ROOT to this StarWhisper checkout if scripts run from the copied skills."
