param(
    [Parameter(Mandatory=$true)]
    [string]$Approach,
    
    [Parameter(Mandatory=$true)]
    [string]$Tolerance,
    
    [Parameter(Mandatory=$true)]
    [string[]]$AtdSizes
)

Write-Host "Running ATD tests for approach: $Approach, tolerance: $Tolerance"
Write-Host "ATD sizes: $($AtdSizes -join ', ')"
Write-Host ""

foreach ($atd in $AtdSizes) {
    $csvName = "${Approach}${atd}.csv"
    $command = "go run .\w_orchestra.go .\utils.go .\working_series_break.go -approach `"$Approach`" -tols `"$Tolerance`" -ratios `"0,10,20,30,40,50,60,70,80,90`" -loops 1000 -atd `"$atd`" -csv `"$csvName`""
    
    Write-Host "Running: $command"
    Write-Host "Output file: $csvName"
    Write-Host "ATD Size: $atd"
    Write-Host "----------------------------------------"
    
    Invoke-Expression $command
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Successfully completed ATD size: $atd" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed for ATD size: $atd" -ForegroundColor Red
    }
    
    Write-Host ""
}

Write-Host "All tests completed!" -ForegroundColor Yellow