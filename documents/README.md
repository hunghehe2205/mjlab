# RobustDexGrasp → mjlab (UR5e + RH5-DG2)

Port phương pháp [RobustDexGrasp](https://github.com/zdchan/RobustDexGrasp)
(RaiSim, UR5 + Allegro) sang mjlab với UR5e + RH5-DG2 right hand.

## Tài liệu

| File | Nội dung |
|---|---|
| [01-analysis.md](01-analysis.md) | Phân tích phương pháp gốc theo code + mapping sang hệ của mình |
| [02-phase1-teacher-plan.md](02-phase1-teacher-plan.md) | Todo plan Phase 1: teacher policy (privileged, PPO) |
| [03-phase2-objects-plan.md](03-phase2-objects-plan.md) | Todo plan Phase 2: scale bộ object + curriculum |
| [04-phase3-student-plan.md](04-phase3-student-plan.md) | Todo plan Phase 3: student (single-view, DAgger + LSTM recon) |

## Trạng thái

- [x] Phân tích code gốc (teacher/student env, cfg, train loop, vec-env wrapper)
- [x] Khảo sát hạ tầng mjlab (manager-based task, sensors, variants, rsl_rl)
- [ ] Phase 1 — Teacher
- [ ] Phase 2 — Objects
- [ ] Phase 3 — Student

## Quyết định khung (đã chốt)

1. **Teacher trước, student sau** — Phase 1 chỉ dựng teacher privileged để chứng
   minh phương pháp chạy trên embodiment mới.
2. **Ít object trước, scale sau** — Phase 1 dùng 3–5 object; pipeline convert cả
   bộ ~35 object để ở Phase 2.
3. **Method-faithful, mjlab-idiomatic** — giữ nguyên thuật toán (hand-centric
   distance vectors, IK pre-grasp, delta action 5 Hz, cấu trúc reward) nhưng cài
   bằng manager-based env + rsl_rl của mjlab, vector hóa GPU.
