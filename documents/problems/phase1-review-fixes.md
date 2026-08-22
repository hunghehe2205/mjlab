# Phase 1 review — findings & fixes (post §B)

Một lượt code-review (15 finding, đều verified) trên diff §A+§B. Dưới đây là các
finding đã sửa + cách sửa. Tất cả đã verify bằng test/probe runtime.

## 6 bug hành vi (chặn train) — đã sửa

1. **Repo test sweep fail (`enable_corruption`).** `test_task_configs.py`
   bắt buộc mọi task train có actor `enable_corruption=True`; env mình để
   `False` → fail. Sửa: actor base `enable_corruption=True` (không term nào có
   noise model → no-op, obs teacher vẫn sạch/privileged); play override vẫn tắt.

2. **Ngón tay bị clamp về soft-limit lúc reset.** `reset_joints_by_offset`
   clamp về `soft_joint_pos_limits`; finger default 0 bị đẩy lên ~0.08 rad. Sửa:
   bake `INIT_FINGER_POSE` (cupped, trong soft-limit) vào `init_state.joint_pos`
   → reset land đúng pose. Probe: `max_finger_reset_err = 0.0000`.

3. **Friction bàn 0.2 vô hiệu.** Equal-priority contact lấy element-wise max →
   object default 1.0 override 0.2. Sửa: table geom `priority=1` → friction bàn
   thắng. Probe: `geom_priority[table] == 1`.

4. **Affordance cloud sample mesh gốc, không phải hull.** Sim va chạm bằng convex
   hull; cloud lệch tới ~5mm ở YCB. Sửa: `precompute` sample `mesh.convex_hull`
   (points nằm trên bề mặt ngón tay thật sự chạm). Regenerate 5 npz.

5. ~~**Delta-action không clip.**~~ **ĐẢO Ở LƯỢT 2 — finding gốc sai.** RobustDexGrasp
   coi `actionStd` là *residual gain*, KHÔNG phải cap/step: delta không bị chặn,
   chỉ *absolute target* bị clip về joint limits. Clip delta về ±scale (mình làm
   ban đầu) chặn cả policy mean + lệch PPO ratio. Sửa lại: bỏ `ACTION_CLIP`. Arm
   đã tự clip target qua actuator ctrlrange; §D thêm soft-limit target clip. Probe:
   action=1 → dịch ≈scale; action=50 → tiến tới limit, bounded, no NaN.

6. **Delta re-anchor mỗi substep → authority 20×.** `apply_actions` chạy mỗi
   decimation substep, mà nó anchor `current_pos` mới mỗi lần → target tiến
   `delta`/substep → authority ~20×. Sửa: `RelativeJointPositionAction` anchor
   target 1 lần ở `process_actions` (mỗi control step), giữ cố định qua substep.
   Probe (action=50, clip cap 0.015): ngón chỉ dịch 0.0079 rad/step thay vì ~0.3.

## Cleanup đã sửa

7. **Import-time disk read.** `DexGraspObject.lowest_point` đọc txt lúc build
   registry → thiếu asset là hỏng `import mjlab.tasks`. Sửa: `lowest_point` thành
   `@property` đọc lazy.

8. ~~**Aliasing term cfg actor↔critic.**~~ **ĐẢO Ở LƯỢT 2 — false positive.**
   `ObservationManager._prepare_terms` đã `deepcopy(term_cfg)` mỗi group
   (`observation_manager.py:456`, có NOTE) TRƯỚC khi null noise → term gốc không
   bao giờ bị mutate, kể cả khi actor/critic share object. Fix `dataclasses.replace`
   của mình là thừa → revert về `{**actor_terms}`.

9. **Test assert keyframe thay vì post-reset.** `test_dexgrasp_env` reset một
   `MjData` mới về keyframe rồi assert → mù với bug ở reset event. Sửa: assert
   `env.scene[...].data.root_link_pos_w` (state sống sau reset).

10. **`nconmax`/`njmax` phóng đại.** 400/4000 (~7× lift_cube 55/600). Giảm về
    150/1500 (headroom cho hand nhiều ngón); revisit ở §H khi đo được peak.

## Lượt review 2 (3 agent song song: correctness / silent-failure / style)

Chạy lại trên diff đã sửa để quét nốt phần minor. Kết quả:

- **C1 (đảo #5):** clip delta ±scale là misport → bỏ (xem #5 ở trên).
- **F8 (đảo #8):** aliasing là false positive → revert (xem #8 ở trên).
- **C2 — rl_cfg lệch reference:** align `value_loss_coef=0.5`, `entropy_coef=0.0`,
  `learning_rate=5e-4`, `max_grad_norm=0.5`, `num_steps_per_env=70` (full-episode
  rollout), `std_range=(0.2,1.0)` (min action std 0.2). Docstring khớp reference.
- **C3 — docstring lazy `lowest_point` overclaim:** sửa lại cho đúng (import module
  không cần asset; build env cfg vẫn đọc). Lazy hoá hoàn toàn để §C khi placement
  chuyển sang reset event.
- **C4 — object/finger friction default 1.0 vs reference ~0.8:** hoãn §F (cùng
  reward grasp); randomized friction là domain-randomization của §F.
- **G1/G2 — comment:** nén khối comment 5 dòng; sửa "arm-only keyframe" → "all".
- **F1 — test:** `test_init_finger_pose_valid` giờ check SOFT limits (reset clamp).
- **F3 — test:** `test_surface_points_on_collision_hull` — cloud phải nằm trên hull
  hiện tại (bắt npz stale).
- **F4 — test:** env test assert `obj_z == TABLE_TOP_Z - lowest_point` chính xác +
  ngón đúng `INIT_FINGER_POSE` sau reset (đọc trước khi step).

Bỏ qua (informational, vẫn "loud"): F5 (assert file exists bị strip dưới `-O` nhưng
`MjSpec.from_file` vẫn raise), F6 (`trimesh.load_mesh` trả Scene cho OBJ nhiều group
— đã có test bắt loud).
