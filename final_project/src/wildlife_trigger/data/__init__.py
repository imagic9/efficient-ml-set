"""Dataset acquisition, manifests, audit and preprocessing.

Owns the CCT-20 split manifests, the `cis-val-clean` derived split, the
`cct_empty_train_v1` supplement, and the canonical preprocessing shared with the
C++ reference implementation via golden fixtures.

The audit here is a hard gate: DESIGN §5.3 stops the project rather than training
around a split problem.
"""
