# Frozen correctness inputs

These files are copied without modification from the 2026-09-12 SysID arm
diagnostic reports. Tests require only NumPy and JAX, not the business project.

- `cholesky_20x20.npz`: float32 SPD matrix, key `M`, shape `(20, 20)`.
  Source: `tmp/npu_cholesky_report_20260912/input.npz`.
  SHA256: `d437a91024b4d6127fc0a4bbb3eb1c2cdd2e4c85d09b09f58fb7345f36a99e25`.
- `collision_support_12x6.npz`: float32 capsule/box face scores, key `support`,
  shape `(12, 6)`. Source: `tmp/cholesky_remaining_20260912/collision/support_input.npz`.
  Expected row argmax: `[4, 4, 4, 5, 3, 3, 3, 4, 5, 5, 0, 5]`.
  SHA256: `f274701b0e4d7282649a080640d0acb48a11903e19fd7a94694649ddd3cfbed0`.

Source paths are relative to
`/data/MJX/SysID_G1_Arm/sysid_g1_arm_delivery/SysID_G1_Real_ArmOnly/`.
