"""Hardware dtype selection, including torch's BF16 emulation default."""
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_object_feature_controls import load_host_class

# Reuse the repository's dependency-light real-host loader: no weights, model
# construction, downloads or CUDA initialization are needed for this function.
preferred_cuda_dtype = load_host_class().maybe_autocast.__globals__["preferred_cuda_dtype"]


class PreferredCudaDtypeTests(unittest.TestCase):
    def test_cpu_uses_fp32_without_querying_cuda_hardware(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False), \
             mock.patch.object(torch.cuda, "get_device_capability") as capability, \
             mock.patch.object(torch.cuda, "is_bf16_supported") as support:
            self.assertEqual(preferred_cuda_dtype(), torch.float32)
            capability.assert_not_called()
            support.assert_not_called()

    def test_v100_uses_fp16_even_when_bf16_emulation_reports_supported(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "get_device_capability", return_value=(7, 0)), \
             mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True):
            self.assertEqual(preferred_cuda_dtype(), torch.float16)

    def test_pre_ampere_hardware_never_uses_the_emulation_result(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)), \
             mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True) as support:
            self.assertEqual(preferred_cuda_dtype(), torch.float16)
            support.assert_not_called()

    def test_ampere_uses_bf16_only_when_hardware_support_probe_succeeds(self):
        def support(*, including_emulation=True):
            return not including_emulation
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)), \
             mock.patch.object(torch.cuda, "is_bf16_supported", side_effect=support) as probe:
            self.assertEqual(preferred_cuda_dtype(), torch.bfloat16)
            probe.assert_called_once_with(including_emulation=False)

    def test_ampere_without_reported_bf16_support_uses_fp16(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)), \
             mock.patch.object(torch.cuda, "is_bf16_supported", return_value=False):
            self.assertEqual(preferred_cuda_dtype(), torch.float16)

    def test_legacy_probe_without_emulation_argument_retains_hardware_gate(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)), \
             mock.patch.object(torch.cuda, "is_bf16_supported", new=lambda: True):
            self.assertEqual(preferred_cuda_dtype(), torch.bfloat16)

    def test_missing_support_api_uses_fp16_even_on_new_hardware(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)), \
             mock.patch.object(torch.cuda, "is_bf16_supported", new=None):
            self.assertEqual(preferred_cuda_dtype(), torch.float16)


if __name__ == "__main__":
    unittest.main()
