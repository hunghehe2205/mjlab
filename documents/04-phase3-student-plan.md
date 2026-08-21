# Phase 3 — Student (single-view + DAgger + LSTM recon)

Điều kiện vào: teacher Phase 2 tốt trên cả bộ object.
Mục tiêu: student chỉ dùng thông tin đo được ngoài đời (single-view cloud,
proprio history, force sensors), distill từ teacher.

## A. Observation student

- [ ] Single-view cloud lúc reset: raycast từ camera (trimesh CPU như gốc,
  hoặc `RayCastSensor`/`CameraSensor` depth + `unproject_depth` của mjlab) →
  cloud tĩnh giữ nguyên cả episode.
- [ ] af_vec student: cdist keypoint ↔ cloud tĩnh (world frame), thay cho
  full-pcd bám object của teacher.
- [ ] History obs: 10 bước × 44+ dim (qpos + sai số PD) qua
  `ObservationGroupCfg(history_length=10)`.
- [ ] Obs delay: `ObservationTermCfg(delay=...)` (gốc: lag 1–2 bước).
- [ ] **Tận dụng force sensor RH5-DG2**: thêm `<sensor>` force cho 6 pad site;
  obs student dùng giá trị đo trực tiếp cho 6 kênh này, LSTM chỉ cần
  reconstruct phần contact per-link còn lại (lợi thế so với Allegro — cân nhắc
  ablation: full recon vs sensor + recon).

## B. Kiến trúc & thuật toán

- [ ] LSTM state-history encoder: 44×10 → latent 26 (recon contact flags +
  impulses), port `LSTM_StateHistoryEncoder`.
- [ ] Distillation runner: dùng `DistillationRunner` của rsl_rl 5.4.2 — viết
  `RslRlDistillationCfg` wrapper cho mjlab (chưa có), truyền qua
  `runner_cls` trong `register_mjlab_task`; nếu thiếu tính năng (mixed
  DAgger+PPO, recon loss) thì tự viết runner theo `dagger_partial.py` gốc.
- [ ] Mixed curriculum: `ppo_ratio = min(iter · 5e-4, 1.0)` — thuần imitation
  → thuần RL trong ~2000 iter; value function dùng obs privileged (teacher
  group); student init từ weights teacher.

## C. Domain randomization (event terms `dr.*`)

- [ ] friction object/finger: {0.5–0.9}.
- [ ] PD gains: tay ±10%, arm ±5% (`dr.pd_gains`).
- [ ] Nhiễu qpos: tay ±0.02 rad, arm ±0.005 rad; nhiễu frame ±0.01 m /
  ±0.02 rad (obs noise cfg).
- [ ] Action delay 1 bước (50%).
- [ ] Perception bias: dịch object ±5 cm một lần khi min-dist tay–object
  < 7 cm (event term interval + điều kiện, port logic `biased`).

## D. Đánh giá

- [ ] Cùng eval script Phase 2 (per-object success, lift script).
- [ ] So teacher vs student trên cùng seed/objects; đo độ rớt success khi
  thêm từng nguồn nhiễu (ablation nhanh).
- [ ] Stress test robustness: `apply_external_force_torque` lên object trong
  khi grasp (paper demo kéo/đẩy vật).

## Ghi chú để sau

- Deploy robot thật (ROS, camera thật, calibration) nằm ngoài phạm vi ba
  phase này — khi cần sẽ lập plan riêng dựa trên `allegro_real/real.py` gốc.
