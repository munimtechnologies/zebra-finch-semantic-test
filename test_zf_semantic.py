# SPDX-License-Identifier: Apache-2.0
# MV2, Munim, Inc. Made with AI agents working in MV2's name.
"""Unit tests for zf_semantic.py (numpy + stdlib only; no data needed)."""

import unittest

import numpy as np

import zf_semantic as z


class TestHelpers(unittest.TestCase):
    def test_rankdata_ties(self):
        np.testing.assert_allclose(z.rankdata([10, 20, 20, 5]), [2, 3.5, 3.5, 1])

    def test_sym_offdiag(self):
        np.testing.assert_allclose(z.sym_offdiag([[5, 1], [3, 7]]), [[0, 2], [2, 0]])

    def test_upper_vec_order(self):
        m = np.array([[10 * i + j for j in range(4)] for i in range(4)], dtype=float)
        np.testing.assert_allclose(z.upper_vec(m), [1, 2, 3, 12, 13, 23])

    def test_partition_counts(self):
        self.assertEqual(len(z.set_partitions(4, (2, 2))), 3)
        self.assertEqual(len(z.set_partitions(8, (1, 1, 2, 2, 2))), 420)
        self.assertEqual(len(z.set_partitions(10, (1, 2, 2, 2, 3))), 12600)
        self.assertEqual(len(z.set_partitions(11, (1, 1, 2, 2, 2, 3))), 69300)

    def test_partitions_unique(self):
        S = z.partition_same_matrix(8, (1, 1, 2, 2, 2))
        self.assertEqual(len({tuple(r) for r in S}), 420)
        self.assertTrue(np.all(S.sum(axis=1) == 3))

    def test_semantic_pairs(self):
        ids = z.group_ids(z.CALL_LIST_DEFAULT, z.SEM_DEFAULT)
        self.assertEqual(int(z.same_vec(ids).sum()), 6)
        self.assertEqual(z.block_sizes(ids), (1, 1, 2, 2, 2, 3))
        labs = [c for c in z.CALL_LIST_DEFAULT if c != "So"]
        ids2 = z.group_ids(labs, z.SEM_DEFAULT)
        self.assertEqual(z.block_sizes(ids2), (1, 2, 2, 2, 3))
        self.assertEqual(int(z.same_vec(ids2).sum()), 6)

    def test_reorder(self):
        idx = z.reorder_index(z.NAME_GRP_DEFAULT, z.CALL_LIST_DEFAULT)
        self.assertEqual([z.NAME_GRP_DEFAULT[i] for i in idx], z.CALL_LIST_DEFAULT)

    def test_acoustic_lor(self):
        L = z.acoustic_lor([[8, 2], [1, 9]])
        self.assertAlmostEqual(L[0, 1], 4.0)
        self.assertAlmostEqual(L[1, 0], np.log2(81.0))
        self.assertEqual(L[0, 0], 0.0)
        L3 = z.acoustic_lor(np.array([[[8, 2], [1, 9]], [[0, 0], [0, 0]]]))
        np.testing.assert_allclose(L3, L)

    def test_auc(self):
        s = np.array([1.0, 0.0, 1.0, 0.0])
        d = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(z.auc_within(d, s), 0.75)
        self.assertAlmostEqual(float(z.auc_all(d, s[None, :])[0]), 0.75)
        d2 = np.array([1.0, 1.0, 3.0, 4.0])
        self.assertAlmostEqual(z.auc_within(d2, s), 0.625)
        self.assertAlmostEqual(float(z.auc_all(d2, s[None, :])[0]), 0.625)

    def test_brier(self):
        self.assertAlmostEqual(z.brier([(0.8, True), (0.25, False), (0.5, None)]),
                               (0.04 + 0.0625) / 2)
        self.assertIsNone(z.brier([(0.5, None)]))


class TestPartialTest(unittest.TestCase):
    def _toy(self):
        ids = [0, 0, 1, 1, 2, 2, 3, 4]
        s = z.same_vec(ids)
        x = np.random.default_rng(1).normal(size=s.size)
        S = z.partition_same_matrix(8, z.block_sizes(ids))
        return s, x, S

    def test_planted_effect_is_unique_max(self):
        s, x, S = self._toy()
        y = x - 10.0 * s
        T_all = z.partial_T(y, x, S)
        T_obs = float(z.partial_T(y, x, s[None, :])[0])
        self.assertAlmostEqual(T_obs, 10.0, places=6)
        self.assertLessEqual(z.exact_p(T_all, T_obs), 2.0 / 420)

    def test_no_semantic_term_gives_p_one(self):
        s, x, S = self._toy()
        y = 2.0 * x + 1.0
        T_all = z.partial_T(y, x, S)
        T_obs = float(z.partial_T(y, x, s[None, :])[0])
        self.assertAlmostEqual(T_obs, 0.0, places=8)
        self.assertEqual(z.exact_p(T_all, T_obs), 1.0)

    def test_scale_free_toy(self):
        ids = [0, 0, 1, 1, 2, 2, 3, 4]
        n = len(ids)
        rng = np.random.default_rng(3)
        A = rng.uniform(1.0, 2.0, size=(n, n))
        A = (A + A.T) / 2.0
        np.fill_diagonal(A, 0.0)
        same = np.array([[1.0 if (ids[i] == ids[j] and i != j) else 0.0
                          for j in range(n)] for i in range(n)])
        B = A - 2.0 * same
        S = z.partition_same_matrix(n, z.block_sizes(ids))
        r = z.scale_free_tests(B, A, ids, S, np.random.default_rng(0))
        self.assertEqual(r["n_partitions"], 420)
        self.assertEqual(r["n_same_pairs"], 3)
        self.assertLessEqual(r["partial_p"], 0.05)
        self.assertEqual(r["auc_beh"], 1.0)
        self.assertGreater(r["auc_beh"], r["auc_ac"])
        self.assertIn("null_fpr_05", r["calibration"])


class TestNotebookPatching(unittest.TestCase):
    def test_clean_source(self):
        src = ("%matplotlib inline\n!pip install x\nx = 5 % 3\nprint('a %d'\n% (1,))\n"
               "get_ipython().run_line_magic('a','b')")
        out = z.clean_source(src)
        lines = out.split("\n")
        self.assertTrue(lines[0].startswith("pass"))
        self.assertTrue(lines[1].startswith("pass"))
        self.assertEqual(lines[2], "x = 5 % 3")
        self.assertEqual(lines[4], "% (1,))")
        self.assertNotIn("get_ipython", out)
        compile(out, "<t>", "exec")

    def test_patch_counts(self):
        src = ("a\n    " + z.TYPO_OLD + "\nb\n    " + z.TYPO_OLD + "\n    " + z.REC_OLD + "\n")
        counts = {}
        out = z.apply_patches(src, z.ARMS["fixed"], counts)
        self.assertEqual(counts[z.TYPO_OLD], 2)
        self.assertEqual(counts[z.REC_OLD], 1)
        self.assertEqual(out.count(z.TYPO_NEW), 2)
        self.assertNotIn(z.TYPO_OLD, out)
        self.assertIn("_BOOT_REC.append", out)

    def test_record_patch_executes(self):
        code = ("import numpy as np\n_BOOT_REC = []\nclass M:\n"
                "    def fit_transform(self, d):\n        return d\n"
                "mds = M()\nconfMatDist = np.eye(2)\nconfMatDistAcoust = np.ones((2, 2))\n"
                "for i in range(2):\n    " + z.REC_OLD + "\n")
        ns = {}
        exec(z.apply_patches(code, z.ARMS["asis"], {}), ns)
        self.assertEqual(len(ns["_BOOT_REC"]), 2)

    def test_noshift_patch(self):
        counts = {}
        out = z.apply_patches("    " + z.ACSHIFT_OLD + "\n", z.ARMS["noshift"], counts)
        self.assertEqual(counts[z.ACSHIFT_OLD], 1)
        self.assertIn("0.0*minval", out)


class TestHeadline(unittest.TestCase):
    def test_headline(self):
        self.assertEqual(z.headline(False, True, 0.99, 0.01), "LOADER_STOP")
        self.assertEqual(z.headline(True, True, None, 0.01), "LOADER_STOP")
        self.assertEqual(z.headline(True, False, 0.99, 0.01), "NOT_REPRODUCED")
        self.assertEqual(z.headline(True, True, 0.99, 0.01), "SURVIVES_BOTH")
        self.assertEqual(z.headline(True, True, 0.90, 0.01), "TYPO_DEPENDENT_SEMANTIC_HOLDS")
        self.assertEqual(z.headline(True, True, 0.90, 0.20), "TYPO_DEPENDENT_NOT_SUPPORTED")
        self.assertEqual(z.headline(True, True, 0.99, 0.20), "Q_SURVIVES_PARTIAL_FAILS")


if __name__ == "__main__":
    unittest.main()
