# Introduction

This is the code for protecting privacy using single-party and multi-party homomorphic encryption via Lattigo Library.

The proprocessed data has been put into the example\dataset folder.

The code is running on Windows platform, and not tested on Linux or Mac.

### Testing performance among different block size choices
```bash
$ go run .\blocksize_choices\test_blocks.go
```

### Testing privacy metrics like entropy, transition
```bash
$ go run .\privacy_metrics\test_metrics.go
```

### Testing attack success rate based on entropy or transition
```bash
$ go run .\asr\test_asr.go 1 2 0 1 60 80
```
Params meaning: strategy, dataset, uniqueATD, target, encryptionRatio, maxHouseholdsNumber

### Testing attack success rate based on data uniqueness
```bash
$ go run .\asr_uniqueness\test_uniqueness.go 1 2 0 1 60 20
```
Params meaning: strategy, dataset, uniqueATD, target, encryptionRatio, maxHouseholdsNumber