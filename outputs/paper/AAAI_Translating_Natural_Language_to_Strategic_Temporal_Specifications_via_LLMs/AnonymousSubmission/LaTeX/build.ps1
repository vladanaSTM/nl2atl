<#
.SYNOPSIS
    Builds anonymous_main.pdf for the AAAI submission.

.DESCRIPTION
    sections/ and bibliography.bib live two directory levels above this
    folder, so TEXINPUTS/BIBINPUTS are pointed up to resolve
    \input{sections/...} and \bibliography{bibliography}. Runs the full
    pdflatex -> bibtex -> pdflatex -> pdflatex sequence.

.EXAMPLE
    .\build.ps1
    .\build.ps1 -Clean
#>
[CmdletBinding()]
param(
    [string]$Main = "anonymous_main",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

# Always run from the folder this script lives in.
Set-Location -Path $PSScriptRoot

# Let TeX/BibTeX also search two levels up (where sections/ and
# bibliography.bib are). Trailing ';' keeps MiKTeX's default search paths.
$env:TEXINPUTS = "..\..;" + $env:TEXINPUTS
$env:BIBINPUTS = "..\..;" + $env:BIBINPUTS

function Invoke-Step {
    # NOTE: do not name this parameter $Args -- that is a reserved PowerShell
    # automatic variable and the array would not bind.
    param([string]$Exe, [string[]]$ArgList)
    Write-Host ">> $Exe $($ArgList -join ' ')" -ForegroundColor Cyan
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe failed with exit code $LASTEXITCODE"
    }
}

$pdfArgs = @("-interaction=nonstopmode", "-halt-on-error", $Main)

Invoke-Step pdflatex $pdfArgs
Invoke-Step bibtex   @($Main)
Invoke-Step pdflatex $pdfArgs
Invoke-Step pdflatex $pdfArgs

if ($Clean) {
    Write-Host ">> cleaning auxiliary files" -ForegroundColor Cyan
    Remove-Item -ErrorAction SilentlyContinue `
        "$Main.aux", "$Main.bbl", "$Main.blg", "$Main.log", "$Main.out"
}

Write-Host "Done: $PSScriptRoot\$Main.pdf" -ForegroundColor Green
