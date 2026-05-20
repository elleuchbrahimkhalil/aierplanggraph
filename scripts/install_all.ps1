<#
Run all install scripts (Windows PowerShell)
Usage: .\scripts\install_all.ps1
#>
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Invoke-Step {
	param(
		[string]$Label,
		[string]$ScriptPath
	)

	Write-Host "Running $Label..."
	try {
		& $ScriptPath
		if (-not $?) {
			Write-Host "$Label failed; continuing"
		}
	} catch {
		Write-Host "$Label failed; continuing"
	}
}

Invoke-Step -Label "install_python.ps1" -ScriptPath "$scriptDir\install_python.ps1"
Invoke-Step -Label "install_frontend.ps1" -ScriptPath "$scriptDir\install_frontend.ps1"

Write-Host "All done. Review output for errors."
