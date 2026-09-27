import importlib.util
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


taxonomy = load_module("taxonomy_release", "scripts/taxonomy_exclude_dataset_sensitivity.py")
gaussian = load_module("gaussian_release", "scripts/reaggregate_analytic_gaussian_null.py")
p300 = load_module("p300_release", "pipelines/run_p300_paired_audit.py")
seediv = load_module("seediv_release", "pipelines/run_seediv_expanded_audit.py")
shards = load_module("shards_release", "hsrg/checkpoint_shards.py")
seediv_validator = load_module("seediv_validator_release", "scripts/validate_seediv_audit.py")


class TaxonomyGuardTests(unittest.TestCase):
    def test_empty_majority_is_undefined(self):
        self.assertIsNone(taxonomy.majority([]))

    def test_unknown_dataset_and_duplicate_joined_keys_fail(self):
        joined = [{"dataset": "known", "arch": "a", "mode": "true", "seed": "1", "layer": "l", "epoch": "0", "hsdd_odi": "0.1", "val_acc": "0.2"}]
        mapping = [{"dataset": "known", "arch": "a", "mode": "true", "seed": "1", "layer": "l", "operation_group": "op", "taxonomy": "tax"}]
        with self.assertRaises(ValueError):
            taxonomy.build_trajectories(joined, mapping, ("unknown", "x", "y"))
        joined.extend([
            {"dataset": "x", "arch": "a", "mode": "true", "seed": "1", "layer": "l", "epoch": "0", "hsdd_odi": "0.1", "val_acc": "0.2"},
            {"dataset": "y", "arch": "a", "mode": "true", "seed": "1", "layer": "l", "epoch": "0", "hsdd_odi": "0.1", "val_acc": "0.2"},
        ])
        mapping.extend([
            {"dataset": "x", "arch": "a", "mode": "true", "seed": "1", "layer": "l", "operation_group": "op", "taxonomy": "tax"},
            {"dataset": "y", "arch": "a", "mode": "true", "seed": "1", "layer": "l", "operation_group": "op", "taxonomy": "tax"},
        ])
        with self.assertRaises(ValueError):
            taxonomy.build_trajectories(joined + [joined[0]], mapping, ("known", "x", "y"))

    def test_strict_bootstrap_rejects_empty_predictions(self):
        with self.assertRaises(ValueError):
            taxonomy.strict_bootstrap([], ("a", "b", "c"), 1, 1)

    def test_invalid_draw_count_is_rejected(self):
        with self.assertRaises(ValueError):
            taxonomy.validate_draws(0)


class GaussianGuardTests(unittest.TestCase):
    def test_altered_alpha_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.csv"
            content = "dataset,arch,mode,seed,epoch,layer,output_dim,actual_raw_odi,analytic_gaussian_alpha_d\\na,a,true,1,0,l,2,0.5,0.123\\n"
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                gaussian.read_compact(path)


class P300PersistenceTests(unittest.TestCase):
    def test_preflight_and_checkpoint_shard_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            present = p300.checkpoint_path(root, "eegnet", 41, 0)
            present.parent.mkdir(parents=True)
            present.write_bytes(b"checkpoint")
            available, missing = p300.preflight_checkpoints(root, ("eegnet",), (41,), (0, 1))
            self.assertEqual(len(available), 1)
            self.assertEqual(len(missing), 1)
            shard = p300.shard_path(root, "eegnet", 41, 0)
            inputs = {"schema": "p300-paired-audit-shard-v2", "cache_sha256": "a", "runtime": {"device": "cpu"}}
            rows = [{"layer": "logits", "cache_val_index": 17}]
            digest = shards.write_verified_shard(shard, inputs=inputs, rows=rows, check={"seed": 41}, manifest={"layers": ["logits"]})
            self.assertIsNotNone(shards.load_verified_shard(shard, digest, expected_rows=1, row_keys={("logits", 17)}, row_key_fields=("layer", "cache_val_index")))

    def test_fingerprint_mismatch_and_partial_shard_are_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shard.json"
            inputs = {"cache_sha256": "a", "selected_indices": [1], "eps": 1e-12, "checkpoint_sha256": "b", "fixed_hooks": ["logits"]}
            digest = shards.write_verified_shard(path, inputs=inputs, rows=[{"row": 1}], check={"ok": True}, manifest={})
            self.assertIsNotNone(shards.load_verified_shard(path, digest))
            changed = dict(inputs, eps=1e-8)
            self.assertIsNone(shards.load_verified_shard(path, shards.fingerprint(changed)))
            path.write_text("{", encoding="utf-8")
            self.assertIsNone(shards.load_verified_shard(path, digest))
            path.write_text("[]", encoding="utf-8")
            self.assertIsNone(shards.load_verified_shard(path, digest))

    def test_p300_actual_cache_index_key_rejects_missing_or_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shard.json"
            inputs = {"schema": "p300-paired-audit-shard-v2", "runtime": {"device": "cpu"}}
            expected = {("layer", 7), ("layer", 11)}
            rows = [{"layer": "layer", "cache_val_index": 7}, {"layer": "layer", "cache_val_index": 11}]
            digest = shards.write_verified_shard(path, inputs=inputs, rows=rows, check={"ok": True}, manifest={})
            self.assertIsNotNone(shards.load_verified_shard(path, digest, expected_rows=2, row_keys=expected, row_key_fields=("layer", "cache_val_index")))
            rows[1]["cache_val_index"] = 7
            shards.write_verified_shard(path, inputs=inputs, rows=rows, check={"ok": True}, manifest={})
            self.assertIsNone(shards.load_verified_shard(path, digest, expected_rows=2, row_keys=expected, row_key_fields=("layer", "cache_val_index")))


class SeedIVResumeTests(unittest.TestCase):
    def _rows(self):
        base = {
            "arch": "eegnet", "seed": "41", "epoch": "0", "checkpoint_relpath": "checkpoint.pt", "checkpoint_sha256": "digest",
            "raw_odi": "0.5", "raw_signed_mean": "0.5", "anchored_odi": "0.25", "anchored_signed_mean": "0.25",
            "anchor_active_bands": "4", "anchor_excluded_bands": "0", "anchor_excluded_pairs": "0", "baseline_sq_over_raw_den": "0.2", "cross_over_raw_den": "0.1", "residual_dot_over_raw_den": "0.2", "signed_reconstruction_max_abs_error": "0", "response_norm_mean": "1", "residual_norm_mean": "1",
        }
        return [dict(base, layer="a", window_position="0", cache_val_index="9"), dict(base, layer="a", window_position="1", cache_val_index="10")]

    def test_seediv_exact_rows_reject_removed_duplicate_and_invalid_anchor_nan(self):
        rows = self._rows()
        options = {"arch": "eegnet", "seed": 41, "epoch": 0, "names": ("a",), "picked": np.asarray([9, 10]), "checkpoint_relpath": "checkpoint.pt", "checkpoint_sha256": "digest"}
        self.assertTrue(seediv.valid_resume_rows(rows, **options))
        self.assertFalse(seediv.valid_resume_rows(rows[:1], **options))
        duplicate = [rows[0], dict(rows[0])]
        self.assertFalse(seediv.valid_resume_rows(duplicate, **options))
        invalid = self._rows()
        invalid[0]["anchored_odi"] = "nan"
        self.assertFalse(seediv.valid_resume_rows(invalid, **options))

    def test_seediv_metadata_reuse_requires_stored_inputs_runtime_and_csv_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shard = root / "shard.csv"
            rows = self._rows()
            with shard.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            inputs = {"schema": "seediv-expanded-audit-shard-v2", "runtime": {"torch": "x", "device": "cuda:0"}}
            metadata = {"resume_inputs": inputs, "resume_fingerprint": shards.fingerprint(inputs), "csv_sha256": seediv.sha256(shard)}
            metadata_path = root / "metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            options = {"resume_inputs": inputs, "arch": "eegnet", "seed": 41, "epoch": 0, "names": ("a",), "picked": np.asarray([9, 10]), "checkpoint_relpath": "checkpoint.pt", "checkpoint_sha256": "digest"}
            self.assertTrue(seediv.resume_metadata_matches(metadata, shard, self._rows(), **options))
            changed_runtime = dict(inputs, runtime={"torch": "changed", "device": "cuda:0"})
            self.assertFalse(seediv.resume_metadata_matches(metadata, shard, self._rows(), **dict(options, resume_inputs=changed_runtime)))
            changed_stored_inputs = dict(metadata, resume_inputs={"schema": "seediv-expanded-audit-shard-v2", "runtime": {"torch": "changed", "device": "cuda:0"}})
            self.assertFalse(seediv.resume_metadata_matches(changed_stored_inputs, shard, self._rows(), **options))
            shard.write_text(shard.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            self.assertFalse(seediv.resume_metadata_matches(metadata, shard, self._rows(), **options))
            metadata_path.write_text("[]", encoding="utf-8")
            self.assertIsInstance(json.loads(metadata_path.read_text(encoding="utf-8")), list)


class PublicInferenceModelTests(unittest.TestCase):
    def test_architecture_constructors_accept_synthetic_p300_shape(self):
        from train_bci2a_arch_layer_audit import arch_spec

        for architecture in p300.FIXED_HOOKS:
            model, _layers, _ = arch_spec(architecture, 8, 128, n_outputs=2, sfreq=128)
            output = model(__import__("torch").zeros((2, 8, 128)))
            self.assertEqual(tuple(output.shape), (2, 2))


class SeedIVCompactValidatorTests(unittest.TestCase):
    def test_compact_validator_rejects_exact_key_miss(self):
        source = ROOT / "results" / "seediv"
        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            coverage = temp_root / "coverage.csv"
            fixed_hooks = temp_root / "fixed.csv"
            summary = temp_root / "summary.json"
            coverage.write_text((source / "checkpoint_coverage.csv").read_text(encoding="utf-8"), encoding="utf-8")
            summary.write_text((source / "expanded_audit_summary.json").read_text(encoding="utf-8"), encoding="utf-8")
            with (source / "fixed_hook_seed_layer_checkpoint_summary.csv").open(encoding="utf-8") as handle:
                hook_rows = list(csv.DictReader(handle))
            with fixed_hooks.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(hook_rows[0]))
                writer.writeheader()
                writer.writerows(hook_rows[1:])
            arguments = ["validate_seediv_audit.py", "--summary", str(summary), "--coverage", str(coverage), "--fixed-hooks", str(fixed_hooks)]
            with patch.object(sys, "argv", arguments):
                with self.assertRaises(ValueError):
                    seediv_validator.main()
