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

## 2. Self-collision probe — HOÃN

Bản gốc (`SampleGraspPose`) resample pregrasp tới khi **không self-collision** bằng
một static probe compile riêng `get_spec_arm_hand_collision()` (chỉ cặp arm↔hand va
chạm, không world) — `train.py:580-615`, đọc 4 flag contact arm trong global_state.

RH5-DG2 **chưa có spec collision arm↔hand riêng**. Muốn probe cần một collision
scheme bật self-contact arm↔hand (contype/conaffinity hoặc contact-pair tường minh),
vì menagerie arm mặc định tắt self-collision link kề.

**Quyết định:** reset event §C **tạm không reject** pregrasp self-collision. Lý do
chấp nhận được ở Phase 1:
- `INIT_FINGER_POSE` đã verify collision-free ở §A.
- Pregrasp side-grasp từ IK (grasp-center cách object 0.25m) thường tay ở ngoài,
  ít khi tay đâm arm.
- IK infeasible đã có fallback (generate_pregrasp trả None → `fallback_arm_qpos`).

**Cần làm khi:** viewer thấy tay xuyên arm/bàn lúc reset → thêm
`get_spec_arm_hand_collision()` cho ur5e_rh5dg2 + probe resample (port
`SelfCollisionProbe`). Contact sensor arm ở §E cũng có thể dùng để reject sau reset.

## 3. Đã xử lý (không phải vấn đề mở)

- **DH residual 0.5mm**: IK dùng DH fit theo MJCF (không textbook), `_ARG_TOL=1e-2`
  để không loại nhầm pose reachable. Sai số FK round-trip ≤0.5mm — thừa cho pre-grasp
  0.25m (policy tinh chỉnh 70 bước). Xem `pregrasp/ik_ur5e.py`.
