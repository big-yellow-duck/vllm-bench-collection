# Qwen3.8-27B MXFP4 on gfx1201

The current TP2/NCCL Simple comparison is [default versus refined FlyDSL](../../../comparisons/gfx1201/qwen3.8-27b-mxfp4/tp2-simple-default-vs-native-20261103/REPORT.md), refreshed 2026-11-05 with full two-pass simple-agent runs.

[Raw TP2 reports](tp2-simple-20261105/) include both passes, source fingerprints and runtime versions. The default uses checkpoint A4 activations; FlyDSL uses FP8 activations, so these are end-to-end serving comparisons rather than precision-matched kernel measurements.
