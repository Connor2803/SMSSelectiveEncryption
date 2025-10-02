# PowerShell script to run fuzzing tests with different percentages
# Runs fuzzing tests in 5% increments from 10-70%, skipping 25% and 50%

# Define the base command components
$baseCommand = "go run .\w_orchestra.go .\utils.go .\working_series_break.go"
$fixedParams = "-approach sliding -tols `"0.05`" -ratios `"0,10,20,30,40,50,60,70,80,90`" -loops 1000 -atd 4 -match 100"

# Define the percentages to test (0-100 in 10% increments)
$percentages = @(0,10,20,30,40,50,60,70,80,90,100)

Write-Host "Starting fuzzing percentage tests..." -ForegroundColor Green
Write-Host "Testing percentages: $($percentages -join ', ')%" -ForegroundColor Cyan
Write-Host ""

$totalTests = $percentages.Count
$currentTest = 0

foreach ($percentage in $percentages) {
    $currentTest++
    $csvFile = "sliding_water_fuzz_{0:D2}.csv" -f $percentage
    
    Write-Host "[$currentTest/$totalTests] Running test with $percentage% fuzzing..." -ForegroundColor Yellow
    Write-Host "Output file: $csvFile" -ForegroundColor Gray
    
    # Build the full command
    $fullCommand = "$baseCommand $fixedParams -fuzz-pct $percentage -csv `"$csvFile`""
    
    Write-Host "Command: $fullCommand" -ForegroundColor DarkGray
    Write-Host ""
    
    # Execute the command
    $startTime = Get-Date
    $result = Invoke-Expression $fullCommand
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Test $percentage% completed successfully in $($duration.ToString('mm\:ss'))" -ForegroundColor Green
    } else {
        Write-Host "✗ Test $percentage% failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }
    
    Write-Host ("-" * 80) -ForegroundColor DarkGray
    Write-Host ""
}

Write-Host "All fuzzing percentage tests completed!" -ForegroundColor Green
Write-Host ""
Write-Host "Generated CSV files:" -ForegroundColor Cyan
foreach ($percentage in $percentages) {
    $csvFile = "fuzz_{0:D2}_basetest.csv" -f $percentage
    if (Test-Path $csvFile) {
        Write-Host "  ✓ $csvFile" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $csvFile (missing)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "To generate heatmaps, run:" -ForegroundColor Yellow
Write-Host "  python fuzz_heatmap_generator.py" -ForegroundColor White