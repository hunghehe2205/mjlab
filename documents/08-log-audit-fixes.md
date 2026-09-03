# 08 — Đối chiếu thông số với bản gốc + đọc log wandb, và các fix

Đối chiếu trực tiếp với `allegro_teacher` (cfg_reg.yaml, train.py, Environment.hpp,
ppo.py, quantitative_eval.py) và bản `rh5dg2_teacher` trên RaiSim. Log: wandb
`mjlab-dexgrasp`, 8 run từ `8f70fa7` tới `4df343f`.

## 1. Khớp bản gốc (không sửa)

PPO (128×128 lrelu, γ 0.996, λ 0.95, 4 epoch, clip 0.2, entropy 0, lr 5e-4, KL 0.01,
std per-dim sàn 0.2), 10/13 reward coeff, cấu trúc weight, divisor 16, clip impulse
0.1/0.2, ngưỡng 0.001/0.01, công thức vel/qvel, delay 50%, sampling vật, không clip
reward, không normalize obs/reward, bootstrap ở time-out.

## 2. Đọc log

| Run | Commit | Env | Iter | contact/bước | dist/bước | LR cuối | std cuối |
|---|---|---|---|---|---|---|---|
| efsqsbb2 | fe072b1 | 4096 | 154 | 0.019 → 0.082 tăng đơn điệu | −1.31 → −1.43 | 4e-3 | 0.98 |
| 4jm4kn3t | 325ec45 | 88, 4 mb | 12 | — | — | — | — |
| advid9la | 325ec45 | 352, 16 mb | 2727 | 0.028 → 0.002 | −1.31 → −1.65 | 1e-5 | 0.41 |
| vcb6edm2 | 580c51b | 352 | 2912 | 0.015 → 0.044 → 0.022 | −1.28 → −1.62 | 1e-5 | 0.30 |
| z3yucft9 | ede9a0a | 352 | 412 | 0 suốt | −2.07 → −1.40 | 2e-3 | 0.94 |
| 4t1vj5xc | 4df343f | 352 | 745 | 0 → 0.016 → 0.004 | −1.90 → −0.94 | 1e-5 | 0.85 |

(`Episode_Reward` của mjlab = tổng episode ÷ 14 s; ×0.2 ra reward/bước.)

- Cấu hình gốc 88 env / 4 minibatch chỉ chạy 12 iteration rồi bị thay; chưa từng được thử.
- LR adaptive với 16 minibatch × 4 epoch = 64 lần chỉnh ×/÷1.5 mỗi iteration (gốc: 16
  lần × 1.2). Ba run cuối: LR dính trần 1e-2 trong 30–40 iteration đầu (value loss nổ
  12k → 57k) rồi đảo 1e-2 ↔ 1e-5 mỗi 10–20 iteration, cuối cùng chết ở sàn.
- Run 4096 env (minibatch 17.9k mẫu) là run duy nhất contact tăng đơn điệu.
- Vật bị quăng (`object_angular_speed_max` ≈ 6.3 rad/s mọi episode) dù flag contact ≈ 0.

## 3. Fix

| # | Vấn đề | Fix |
|---|---|---|
| 1 | `sensor_impulse` nhân `mean(force) × sim_dt`; `ac33c3b` viết ở dt 0.01, `f29989d` đổi dt 0.005 → impulse còn một nửa, ngưỡng flag thực tế 2 N, clip 20 N | `REFERENCE_IMPULSE_DT = 0.01` cố định, không theo timestep; test chặn hồi quy |
| 2 | Standoff: `APPROACH_DISTANCE = 0.25` là cho tâm lòng bàn tay Allegro; site `grasp_center` ở đây cách đầu ngón ~0.10 m → đầu ngón cách vật 0.14–0.18 m, mất ~30/70 bước tiếp cận. Bản RaiSim rh5dg2 của bạn để ngón cách ~2 cm | Đo 60 reset × 5 vật: ngón gần nhất ≈ d − 0.10. Chọn **0.13**: min 3.3 cm, mean 6.5 cm; trên cả 35 vật (140 reset) min 2.2 cm (solder_iron_head), mean 6.7 cm, 0 reset chạm |
| 3 | `soft_joint_pos_limit_factor = 0.9` cắt ~10% tầm ngón mỗi phía; gốc clamp hard limit | 1.0 |
| 4 | `num_mini_batches = 16` + adaptive → LR bất ổn (§2) | 4 như gốc, `schedule="fixed"` 5e-4 |
| 5 | Eval kẹp action arm ±1 (0.005 rad/bước, tối đa 0.45 rad cả pha lift); gốc không giới hạn | Bỏ clamp, target = pose nội suy; in thêm `track_err` |

Còn lại, chưa sửa: eval vẫn 70 bước grasp + lift IK 0.15 m (gốc 100 bước + nội suy về
HOME 80 bước) nên số không so được với paper; `INIT_FINGER_POSE` ngón cái (yaw 1.2)
lệch bản RaiSim (0.3), cần xem viewer; coeff obj_vel/qvel/disp mềm hơn gốc
(−5/−0.1/−2 vs −15/−0.2/−5) là giả thuyết chưa đối chứng.

## 4. Run kế tiếp

1024 env, 4 minibatch (17.9k mẫu/minibatch, 16 update/iteration như gốc), LR cố định.
Ước tính ~26 s/iteration trên 4060 Ti. Cổng: `affordance_contact` tăng đơn điệu,
`object_angular_velocity` không còn là term phạt lớn nhất, LR không chạm biên.
