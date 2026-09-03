from __future__ import annotations

import unittest

import numpy as np

from sentinel.drift import PSIDriftDetector


class DriftTests(unittest.TestCase):
    def test_psi_distinguishes_stable_and_shifted_batches(self) -> None:
        rng = np.random.default_rng(42)
        reference = rng.normal(0, 1, size=(500, 9))
        stable = rng.normal(0, 1, size=(500, 9))
        shifted = rng.normal(2, 1, size=(500, 9))
        detector = PSIDriftDetector(threshold=0.25)
        detector.fit_reference(reference)

        self.assertFalse(detector.score(stable).detected)
        self.assertTrue(detector.score(shifted).detected)


if __name__ == "__main__":
    unittest.main()

