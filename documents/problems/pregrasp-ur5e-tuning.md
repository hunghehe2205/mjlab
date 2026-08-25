# Pre-grasp §C — giá trị port từ UR5 cần verify cho UR5e + RH5-DG2

Toàn bộ chain IK/kinematics/visibility đã validate bằng test (FK round-trip ≤0.5mm,
grasp-center round-trip sub-mm, generator đặt grasp-center đúng phía camera trên
object thật). Nhưng vài **hằng số port thẳng từ setup UR5+Allegro gốc** chưa được
kiểm bằng viewer/empirical trên embodiment mình. Ghi lại để tune ở §C-verify / §H.

## 1. Cần verify trong viewer

| Hằng | Nơi | Rủi ro |
|---|---|---|
| `CAMERA_POSITION = [0.035, -0.58, 1.531]` | `generator.py` | Port từ `cfg_reg.yaml` gốc. Scene mình bàn tâm y=−0.55, top 0.771 — camera ở y=−0.58, cao 0.76m trên bàn, hợp lý về mặt số, nhưng **hướng tiếp cận** (object→camera) quyết định phía grasp; cần nhìn viewer xem tay có tiếp cận từ hướng tự nhiên không. |
| `grasp_center` site `[0.05, 0, 0.11]` palm-local | `right_hand.xml` / §A | Ước lượng đầu (đã ghi ở §A). T_flange_gc đo được = translation thuần, rotation identity → grasp_center cùng hướng flange. Cần viewer xác nhận site nằm đúng lòng bàn tay. |
| `INIT_FINGER_POSE` (cupped) | `ur5e_rh5dg2_constants.py` | §A đã verify self-collision-free (soft limits). Cần viewer xem có "ôm" object hợp lý không (ring/pinky không có yaw → khác Allegro). |
| `_wrist_penalty` (const 1.57, 3.2) | `generator.py` | Tie-breaker chọn palm-roll, UR5-specific. Score chính là projection width (coeff 5); term wrist chỉ phá hoà. Nhánh IK UR5e khác UR5 → hằng có thể không chuyển. Ảnh hưởng thấp. |
| `fallback_arm_qpos` `[θ+π/2−0.3, −1.57, 1.57, 0, 1.57, −1.57]` | `generator.py` | Pose canonical khi IK fail, port thẳng từ UR5. Cần verify reachable + collision-free trên UR5e; nếu lệch thì chỉnh (fallback hiếm khi kích hoạt vì IK side-grasp thường có nghiệm). |

## 2. Self-collision probe — ĐÃ HIỆN THỰC

Bản gốc (`SampleGraspPose`) resample pregrasp tới khi **không self-collision** bằng
một static probe compile riêng `get_spec_arm_hand_collision()` (chỉ cặp arm↔hand va
chạm, không world) — `train.py:580-615`, đọc 4 flag contact arm trong global_state.

`ArmHandSelfCollisionProbe` compile robot MJCF tĩnh, ghi arm qpos candidate cùng
`INIT_FINGER_POSE`, rồi lọc contact có một body thuộc arm và body kia thuộc subtree
hand. Reset reject candidate đó và sample lại tối đa 8 lần. Sau khi hết lượt,
`fallback_arm_qpos` cũng phải pass probe; nếu không, reset về HOME pose đã kiểm tra
collision-free. Probe không kiểm tra bàn/object, vì pre-grasp tĩnh còn cách object
0.25 m và các collision đó thuộc simulator rollout.

## 2b. Deep-review §C (19 finding) — đợt fix

**6 bug behavioral đã fix + test (đều verify bằng probe trước khi sửa):**
1. `load_affordance_mesh` trả **convex_hull** (trước trả raw → ~6% ray miss vì
   concavity; cloud & MuJoCo collision đều trên hull). Test `is_convex`.
2. `ik_ur5e.closest()` **wrap delta về (−π,π]** trước weighted-sum (trước chọn
   nhầm branch ~4.3%). Test: seed +2π/khớp phải trả đúng branch.
3. `fallback_arm_qpos` re-derive `angle + π − 0.3` (base Rz(180) vs UR5 Rz(90);
   trước gc quay lưng object, az lệch ~94°). Test: gc cùng phía + gần object.
4. `_wrist_penalty` **đối xứng** `abs(|w2|−1.57)` (trước chỉ min tại +1.57, mâu
   thuẫn seed HOME w2=−1.57). Test symmetric.
5. RNG seed từ **global numpy** (mjlab `seed_rng` từ env seed) thay vì raw
   `cfg.seed` (None → OS entropy). Test reproducible với global seed. *Giới hạn
   còn lại:* stream frozen lúc build → `env.reset(seed=N)` sau build KHÔNG reseed;
   chỉ deterministic theo build-time / runner path (chấp nhận, như event mjlab khác).
6. Spawn **+2mm clearance** (như gốc 0.773/0.771) tránh contact t=0.

**Trap fix kèm theo:** `mj_name2id` guard −1 (`_require_id`); `__all__` cho
`mdp/events.py` (chặn star-export lộ np/torch/oc/rc); reset loop batch H2D +
bỏ `.clone()` thừa; bump IK round-trip test 40→100 pose.

**Deferred (không phải bug, có lý do — chờ §E/§F hoặc design pass):**
- **Scene geometry 3 nguồn** (`TABLE_CENTER` / polar band / `CAMERA_POSITION`):
  consolidate về 1 source — design refactor, làm khi wire variant/scene §B[~].
- **object_name không couple scene entity**: gắn với chuyển single→variant.
- **params double-contract** `ResetGraspPose.__call__`: là idiom event mjlab
  (cfg.params đọc ở init), không sửa.
- **elbow soft-limit không clamp** (latent, 0/200 trong band Phase 1): revisit
  Phase 2 khi band rộng hơn.
- **trimesh trên import path (~0.2s)** + **dup `_inv_T`** + **`_quat2mat` DIY**:
  perf/cosmetic, để pass cleanup riêng (scipy quat xyzw≠wxyz → không đổi vội).
- **DH 0.0997/0.0996 vs model 0.1/0.1**: `_E`/`_ARG_TOL` hấp thụ, FK ≤0.5mm —
  informational, không phải bug.

## 3. Đã xử lý (không phải vấn đề mở)

- **DH residual 0.5mm**: IK dùng DH fit theo MJCF (không textbook), `_ARG_TOL=1e-2`
  để không loại nhầm pose reachable. Sai số FK round-trip ≤0.5mm — thừa cho pre-grasp
  0.25m (policy tinh chỉnh 70 bước). Xem `pregrasp/ik_ur5e.py`.
