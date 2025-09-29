# PowerShell script to run 80% threshold matching with varying fuzzing percentages
Write-Host "=== Running 80% Threshold Matching Tests ===" -ForegroundColor Green
Write-Host "Testing fuzzing percentages from 0% to 100% in 10% increments" -ForegroundColor Yellow

$fuzzingPercentages = @(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
$tolerance = 0.05
$encryptionRatios = "0,10,20,30,40,50,60,70,80,90"
$loops = 1000
$atdSize = 4
$approach = "adaptive"

foreach ($fuzzPct in $fuzzingPercentages) {
    $csvFileName = "threshold90_fuzz_${fuzzPct}_test.csv"
    
    Write-Host "Running test with $fuzzPct% fuzzing..." -ForegroundColor Cyan
    Write-Host "  Output: $csvFileName"
    
    try {
        $goArgs = @(
            "run", ".\w_orchestra.go", ".\utils.go", ".\working_series_break.go",
            "-approach", $approach,
            "-tol", $tolerance,
            "-ratios", $encryptionRatios,
            "-fuzz-pct", $fuzzPct,
            "-loops", $loops,
            "-atd", $atdSize,
            "-match", 90,
            "-csv", $csvFileName,
            "-debug", "false"
        )
        
        $startTime = Get-Date
        & "go" @goArgs
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   Test completed in $($duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
        } else {
            Write-Host "   Test failed with exit code $LASTEXITCODE" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "   Error: $_" -ForegroundColor Red
    }
}

Write-Host "=== Tests Completed ===" -ForegroundColor Green
Write-Host "Run: python fuzz_heatmap_generator.py -p 'threshold80_fuzz_#_test.csv' -o 'threshold80_plots'"
