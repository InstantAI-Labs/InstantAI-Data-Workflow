param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$root\src;$root"
& $Command[0] @Command[1..($Command.Length - 1)]
