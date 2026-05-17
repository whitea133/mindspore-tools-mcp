#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mindspore-tools-mcp Core Test Suite

Modules: tools, linter_tools, msutils_tools, api_tools, template_tools
Tests: ~66
Run: pytest tests/test_core.py -v  or  python -m unittest tests.test_core -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---- Mock mindspore (not installable in CI) ----
class _FakeMS:
    def __getattr__(self, name): return _FakeMS()
    def __call__(self, *a, **k): return _FakeMS()
    def __repr__(self): return "<MockMS>"

for _m in ("mindspore", "mindspore.nn", "mindspore.ops", "mindspore.dataset",
           "mindspore.common", "mindspore.train", "mindspore.numpy", "mindspore.model_zoo"):
    if _m not in sys.modules:
        sys.modules[_m] = _FakeMS()


# ---- Module-level imports (avoid self-binding) ----
from mindspore_tools_mcp.tools import (
    list_models, get_model_info, recommend_models, compare_models,
    query_op_mapping, diagnose_translation, fetch_official_models,
)
from mindspore_tools_mcp.linter_tools import (
    lint_mindspore_code, get_lint_rules, lint_code_snippet, compare_code_snippets,
)
from mindspore_tools_mcp.msutils_tools import (
    generate_adversarial_attack, evaluate_model_robustness,
    create_data_augmentation_pipeline, get_lr_scheduler, get_training_callbacks,
    compute_model_complexity, setup_distributed_training,
    quantize_model, convert_model_format,
)
from mindspore_tools_mcp.api_tools import (
    get_api_examples, search_apis, list_api_categories,
    get_related_apis, get_quick_reference,
)
from mindspore_tools_mcp.template_tools import (
    generate_training_template, get_available_options,
    preview_template, generate_quick_start,
)


class TestTools(unittest.TestCase):
    """tools.py: model list/query/recommend/compare/API mapping"""

    def test_list_models_returns_list(self):
        self.assertIsInstance(list_models(), list)

    def test_list_models_with_task_filter(self):
        self.assertIsInstance(list_models(task="image classification"), list)

    def test_list_models_with_q_filter(self):
        self.assertIsInstance(list_models(q="resnet"), list)

    def test_list_models_with_group_filter(self):
        self.assertIsInstance(list_models(group="cv"), list)

    def test_list_models_with_category_filter(self):
        self.assertIsInstance(list_models(category="vision"), list)

    def test_get_model_info_existing(self):
        models = list_models()
        if models:
            mid = models[0].get("id") or models[0].get("name", "resnet50")
            self.assertIsInstance(get_model_info(mid), dict)

    def test_get_model_info_not_found_raises(self):
        with self.assertRaises(ValueError):
            get_model_info("__nonexistent_xyz__")

    def test_recommend_models_returns_dict(self):
        r = recommend_models("image classification")
        self.assertIsInstance(r, dict)
        self.assertIn("recommendations", r)
        self.assertIn("total_found", r)

    def test_recommend_models_with_hardware(self):
        self.assertIsInstance(recommend_models("cv", hardware="ascend", limit=2), dict)

    def test_recommend_models_limit(self):
        r = recommend_models("resnet", limit=2)
        self.assertLessEqual(len(r.get("recommendations", [])), 2)

    def test_compare_models_returns_dict(self):
        r = compare_models(["resnet50", "vit_b_16"])
        self.assertIsInstance(r, dict)
        self.assertIn("models", r)

    def test_compare_models_single_model(self):
        self.assertIsInstance(compare_models(["resnet50"]), dict)

    def test_query_op_mapping_returns_dict(self):
        self.assertIsInstance(query_op_mapping("conv2d"), dict)

    def test_query_op_mapping_with_section(self):
        self.assertIsInstance(query_op_mapping("tensor", section="tensor"), dict)

    def test_diagnose_translation_returns_dict(self):
        r = diagnose_translation("x=torch.tensor([1])", "x=ms.Tensor([1])")
        self.assertIsInstance(r, dict)

    def test_fetch_official_models_returns_dict(self):
        self.assertIsInstance(fetch_official_models(), dict)


class TestLinterTools(unittest.TestCase):
    """linter_tools.py: MindSpore code linting"""

    _GOOD_CODE = (
        "import mindspore as ms\nfrom mindspore import nn\n"
        "class Net(nn.Cell):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Dense(784, 10)\n"
        "    def construct(self, x):\n"
        "        return self.fc(x)\n"
    )

    def test_lint_valid_code(self):
        r = lint_mindspore_code(self._GOOD_CODE)
        self.assertIsInstance(r, dict)
        self.assertIn("score", r)
        self.assertIsInstance(r["score"], (int, float))

    def test_lint_empty_code(self):
        r = lint_mindspore_code("")
        self.assertIsInstance(r, dict)
        self.assertIn("score", r)

    def test_lint_bad_code(self):
        r = lint_mindspore_code("import torch\nx=torch.Tensor([1])")
        self.assertIsInstance(r, dict)
        self.assertIn("score", r)

    def test_lint_style_json(self):
        """JSON output style"""
        r = lint_mindspore_code("x=1", style="json")
        self.assertIsInstance(r, dict)
        self.assertIn("formatted_report", r)

    def test_lint_style_markdown(self):
        r = lint_mindspore_code("x=1", style="markdown")
        self.assertIsInstance(r, dict)
        self.assertIn("formatted_report", r)

    def test_lint_level_strict(self):
        self.assertIsInstance(lint_mindspore_code("x=1", level="strict"), dict)

    def test_lint_level_quick(self):
        self.assertIsInstance(lint_mindspore_code("x=1", level="quick"), dict)

    def test_get_lint_rules_returns_dict(self):
        self.assertIsInstance(get_lint_rules(), dict)

    def test_lint_code_snippet(self):
        self.assertIsInstance(lint_code_snippet("python", "import ms\nx=ms.Tensor([1])"), dict)

    def test_compare_code_snippets(self):
        r = compare_code_snippets(
            "import mindspore as ms\nx=ms.Tensor([1])",
            "import torch as t\nx=t.Tensor([1])",
        )
        self.assertIsInstance(r, dict)


class TestMsutilsTools(unittest.TestCase):
    """msutils_tools.py: attacks/augmentation/training/complexity etc."""

    # --- attacks ---
    def test_attack_fgsm(self):
        r = generate_adversarial_attack("fgsm", epsilon=0.03)
        self.assertEqual(r["attack_type"], "fgsm")
        self.assertIn("config", r)
        self.assertIn("code_example", r)

    def test_attack_pgd(self):
        r = generate_adversarial_attack("pgd", epsilon=0.3, num_iterations=40)
        self.assertEqual(r["attack_type"], "pgd")

    def test_attack_deepfool(self):
        r = generate_adversarial_attack("deepfool")
        self.assertIn("code_example", r)

    def test_attack_cw(self):
        self.assertIsInstance(generate_adversarial_attack("cw", num_iterations=20), dict)

    def test_attack_jsma(self):
        self.assertIsInstance(generate_adversarial_attack("jsma"), dict)

    def test_attack_targeted(self):
        self.assertIsInstance(generate_adversarial_attack("fgsm", target_class=7), dict)

    # --- robustness ---
    def test_evaluate_robustness(self):
        fake = {"name": "test_net", "num_classes": 10, "input_shape": (3, 224, 224)}
        r = evaluate_model_robustness(
            model_info=fake,
            attack_configs=[{"method": "fgsm", "epsilon": 0.01}],
        )
        self.assertIsInstance(r, dict)

    # --- augmentation ---
    def test_augmentation_image(self):
        r = create_data_augmentation_pipeline(
            task_type="image_classification",
            augmentations=["random_crop", "flip"],
        )
        self.assertIsInstance(r, dict)
        self.assertTrue("augmentations" in r or "pipeline" in r)

    def test_augmentation_nlp(self):
        self.assertIsInstance(create_data_augmentation_pipeline(
            task_type="nlp", augmentations=["random_delete"]
        ), dict)

    # --- lr scheduler ---
    def test_lr_cosine(self):
        r = get_lr_scheduler(scheduler_type="cosine_annealing", total_epochs=90, base_lr=0.001)
        self.assertIsInstance(r, dict)
        self.assertTrue("config" in r or "schedule_points" in r)

    def test_lr_step(self):
        self.assertIsInstance(get_lr_scheduler(scheduler_type="step", total_epochs=90), dict)

    def test_lr_polynomial(self):
        self.assertIsInstance(get_lr_scheduler(scheduler_type="polynomial_decay", total_epochs=90), dict)

    # --- callbacks ---
    def test_callbacks_default(self):
        r = get_training_callbacks()
        self.assertIsInstance(r, dict)
        self.assertTrue("callbacks" in r or "code_example" in r)

    def test_callbacks_with_checkpoint(self):
        self.assertIsInstance(get_training_callbacks(
            checkpoint_config={"save_freq": 5}
        ), dict)

    # --- complexity ---
    def test_complexity_compute(self):
        r = compute_model_complexity(model_name="resnet50", input_shape=(3, 224, 224))
        self.assertIsInstance(r, dict)
        # keys vary: config/flops/params/etc.
        self.assertTrue(len(r) > 0)

    # --- distributed ---
    def test_distributed_setup(self):
        r = setup_distributed_training(backend="nccl", num_gpus=8)
        self.assertIsInstance(r, dict)
        self.assertTrue("config" in r or "launch_command" in r)

    # --- quantization ---
    def test_quantization(self):
        r = quantize_model(quantization_type="dynamic", precision="int8")
        self.assertIsInstance(r, dict)
        self.assertTrue("config" in r or "code_example" in r)

    # --- conversion ---
    def test_conversion_air_to_mindir(self):
        self.assertIsInstance(convert_model_format(source_format="air", target_format="mindir"), dict)


class TestApiTools(unittest.TestCase):
    """api_tools.py: API example search"""

    def test_get_api_examples_conv2d(self):
        r = get_api_examples("nn.Conv2d")
        self.assertIsInstance(r, str)
        self.assertTrue(len(r) > 0)

    def test_search_apis_tensor(self):
        self.assertIsInstance(search_apis("tensor", max_results=5), str)

    def test_search_apis_loss(self):
        self.assertIsInstance(search_apis("loss", max_results=5), str)

    def test_list_api_categories_not_empty(self):
        r = list_api_categories()
        self.assertIsInstance(r, str)
        self.assertTrue(len(r) > 0)

    def test_get_related_apis(self):
        self.assertIsInstance(get_related_apis("nn.Conv2d"), str)

    def test_get_quick_reference(self):
        self.assertIsInstance(get_quick_reference("nn.Dense"), str)


class TestTemplateTools(unittest.TestCase):
    """template_tools.py: training template generation"""

    def test_generate_template_image_classification(self):
        r = generate_training_template(
            task="image_classification", model="resnet50",
            dataset="cifar10", num_epochs=10, batch_size=32,
        )
        self.assertIsInstance(r, dict)
        self.assertIn("script", r)
        self.assertGreater(len(r["script"]), 100)

    def test_generate_template_object_detection(self):
        r = generate_training_template(task="object_detection", model="yolov5",
                                        dataset="coco", num_epochs=5)
        self.assertIn("script", r)

    def test_generate_template_nlp(self):
        r = generate_training_template(task="nlp", model="bert",
                                        dataset="imdb", num_epochs=3)
        self.assertIn("script", r)

    def test_generate_template_gpu(self):
        self.assertIsInstance(generate_training_template(
            task="image_classification", model="resnet18",
            dataset="cifar10", hardware="GPU"), dict)

    def test_generate_template_cpu(self):
        self.assertIsInstance(generate_training_template(
            task="image_classification", model="lenet",
            dataset="mnist", hardware="CPU"), dict)

    def test_generate_template_no_amp(self):
        self.assertIsInstance(generate_training_template(
            task="image_classification", model="resnet50",
            dataset="cifar10", use_amp=False), dict)

    def test_generate_template_sgd_optimizer(self):
        self.assertIsInstance(generate_training_template(
            task="image_classification", model="resnet50",
            dataset="cifar10", optimizer="sgd"), dict)

    def test_get_available_options(self):
        opts = get_available_options()
        self.assertIsInstance(opts, dict)
        for k in ("tasks", "models", "datasets", "hardware", "optimizers"):
            self.assertIn(k, opts)

    def test_preview_template(self):
        r = preview_template(task="image_classification", model="resnet50")
        self.assertIsInstance(r, dict)
        self.assertTrue("preview" in r or "script" in r or "header" in r)

    def test_quick_start(self):
        r = generate_quick_start()
        self.assertIsInstance(r, dict)
        self.assertTrue("script" in r or "guide" in r or "config" in r)


class TestEdgeCases(unittest.TestCase):
    """Edge cases and error handling"""

    def test_import_package(self):
        import mindspore_tools_mcp
        self.assertTrue(hasattr(mindspore_tools_mcp, "tools"))

    def test_recommend_empty_query(self):
        self.assertIsInstance(recommend_models("", limit=1), dict)

    def test_recommend_special_chars(self):
        self.assertIsInstance(recommend_models("!@#$%", limit=1), dict)

    def test_lint_unicode(self):
        self.assertIsInstance(lint_mindspore_code("# comment\nx=1"), dict)

    def test_template_minimal_params(self):
        r = generate_training_template()
        self.assertIsInstance(r, dict)
        self.assertIn("script", r)

    def test_attack_unknown_type(self):
        try:
            self.assertIsInstance(generate_adversarial_attack("unknown_xyz"), dict)
        except (ValueError, KeyError):
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
