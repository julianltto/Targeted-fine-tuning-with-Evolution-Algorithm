"""MathNeuro 重构后的核心模块。

把原 `MathNeuro.py` 里散乱的剪枝逻辑(find_params / find_good_params / prune / scale)
合并、改名、加注释后放在 core.py 里。外面只要 `from mathneuro.core import ...` 就行。
"""
from mathneuro.core import (
    register_activation_hooks,
    remove_hooks,
    make_math_prompt_fn,
    make_calibration_prompt_fn,
    compute_importance,
    top_k_mask,
    build_prune_mask,
    apply_mask_to_model,
)

__all__ = [
    "register_activation_hooks",
    "remove_hooks",
    "make_math_prompt_fn",
    "make_calibration_prompt_fn",
    "compute_importance",
    "top_k_mask",
    "build_prune_mask",
    "apply_mask_to_model",
]
