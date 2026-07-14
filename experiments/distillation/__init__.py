"""Causal Intervention Distillation (CID) pipeline.

Phase 1 (analysis, no GPU): rescue_stats, plots, analyze_rescue, archive
Phase 2 (mining):           mining
Phase 3 (training):         datasets, train_lora
Phase 4 (eval):             eval_held_out
"""
