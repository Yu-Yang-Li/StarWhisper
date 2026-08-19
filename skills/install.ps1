<#
.SYNOPSIS
Install the StarWhisper skill pack into Codex and/or Cursor.

.EXAMPLE
powershell -File skills/install.ps1 -List
powershell -File skills/install.ps1
powershell -File skills/install.ps1 -Set native -Target codex
#>
[CmdletBinding()]
param(
    [ValidateSet('all', 'native', 'research')]
    [string]$Set = 'all',

    [ValidateSet('both', 'codex', 'cursor')]
    [string]$Target = 'both',

    [switch]$List,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here

$skills = Get-ChildItem -LiteralPath $here -Directory |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'SKILL.md') } |
    Sort-Object Name

switch ($Set) {
    'native'   { $skills = $skills | Where-Object { $_.Name -like 'starwhisper-*' } }
    'research' { $skills = $skills | Where-Object { $_.Name -notlike 'starwhisper-*' } }
}

if (-not $skills) { throw "no skills matched -Set $Set under $here" }

if ($List) {
    foreach ($s in $skills) {
        $kind = if ($s.Name -like 'starwhisper-*') { 'native  ' } else { 'research' }
        Write-Host ("{0}  {1}" -f $kind, $s.Name)
    }
    Write-Host ("{0} skills" -f $skills.Count)
    return
}

$targets = @()
if ($Target -in @('both', 'codex'))  { $targets += Join-Path $env:USERPROFILE '.codex\skills' }
if ($Target -in @('both', 'cursor')) { $targets += Join-Path $env:USERPROFILE '.cursor\skills' }

foreach ($root in $targets) {
    if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $root | Out-Null }
    foreach ($s in $skills) {
        $dest = Join-Path $root $s.Name
        if ($DryRun) {
            Write-Host "would install $($s.Name) -> $dest"
            continue
        }
        if (Test-Path -LiteralPath $dest) { Remove-Item -LiteralPath $dest -Recurse -Force }
        Copy-Item -LiteralPath $s.FullName -Destination $dest -Recurse
        Write-Host "installed $($s.Name) -> $dest"
    }
}

if ($DryRun) { return }

Write-Host ''
Write-Host "Installed $($skills.Count) skills into $($targets.Count) location(s)."
Write-Host "Point the native skills at this checkout so they can read snclock/, explore/ and NGSS/:"
Write-Host "  setx STARWHISPER_ROOT `"$repo`""
Write-Host "Skills with extra Python deps: experiment-design, statistical-analysis, thesis-audit-reviewer, visual-deck-builder, papercheck."
Write-Host "The four starwhisper-* skills are stdlib only and need nothing installed."
