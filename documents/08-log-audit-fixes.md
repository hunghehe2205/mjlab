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

## 5. Run 4gd2kunn (1024 env, RTX 3060, a06258f): ngón cái chạm "miễn phí"

Contact tăng đơn điệu 0 → 1.54 trong 28 iter (gấp 10 run tốt nhất cũ), ~28 s/iter.
Nhưng rollout `model_50.pt` (64 env, potted_meat_can) cho thấy policy khai thác pose
ngón cái: thumb yaw 1.2 đưa đầu ngón cái ra trước các ngón khác 4 cm trên tia tiếp
cận; ngón cái chạm ở bước ~5 trong 100% episode, các ngón khác bước 15–18;
78–81% contact reward đến từ ngón cái (trọng số thumb dip = 12/36). Thumb yaw không
đổi cả episode, các ngón khác gần như không gập. Quét standoff tại reset
(potted_meat_can, 64 env): yaw 1.2 chạm từ d = 0.08 (chỉ ngón cái, 0.37–0.44
reward/bước không cần gập ngón); yaw 0.3 (bản RaiSim của mình) không chạm tới
d = 0.02, các đầu ngón ngang nhau.

Fix: `INIT_FINGER_POSE` thumb yaw 0.3, mcp 0.2 (đúng bản RaiSim); vì ngón cái không
còn nhô ra, `APPROACH_DISTANCE` 0.13 → 0.10 (35 vật, 140 reset: min 3.0 cm, mean
7.9 cm, 0 chạm). Run lại từ đầu.

## 6. Run 8yelo0qc (stage1 gate, 5fce3aa): bão hoà, obs normalize, metric nắm

Iter 1724, ~28 s/iter. Reward 54.5 (400) → 66 (1000) → 81 (1400) → 75 ± 3 (300 iter
cuối); contact/impulse phẳng 4.25/2.95 từ 1400; std 1.0 → 0.48 đơn điệu; LR cố định.
PPO ổn. Nhưng object_displacement −0.31 (700) → −0.88 (1600), action arm 1.2 → 1.8, từ
1200 có 1–4% episode vật văng khỏi workspace: policy ép/kéo vật, không nắm tốt hơn.
Eval lift: 35% (400), 0% (700), 62% (1000), 23% (1100), 56–58% (1200–1400).

Rollout model_700 vs model_1000 (64 env, potted_meat_can): cả hai qua gate ~100% bước
từ bước 15, contact reward/bước 0.74 vs 0.82, nhưng 700 là "vuốt móc" (thumb yaw 0.10
nằm cạnh ngón trỏ, mcp 0 / pip 0.1 / dip 0.9, Σ|xy| 52 N, tuột khi nâng) còn 1000 là
"bao trọn" (yaw 0.82, pip 0.6 / dip 0.5, Σ|xy| 163 N, mang đúng 2.4 N khi nâng).
Reward bão hoà (flag đầy, impulse chạm clip) nên phẳng trên cả họ; eval chỉ chụp
policy đang trôi ở đâu. Giả thuyết "gate nhị phân là vách reward" bị loại: train trơn
tại 700/1100, không env nào ở biên gate. Bản gốc train không có lift (`lift_steps = 0`,
chỉ bật khi eval) mà vẫn 94.6%: khớp đối ngón Allegro 0.26–1.40 rad không có pose
không đối lực, thumb yaw RH5-DG2 0–1.92 thì có.

Obs (đo model_1000, 191 chiều): joint_pos std 1.03 chiếm 51% Σ|x|, wrist 1.04,
contacts 0.38, af_vec 0.027 (3.5%), pd_error 0.022; 22 chiều gần hằng. Value loss
40–700 suốt run (300 iter cuối 134, return ~75). Bản gốc không normalize (§1 đúng,
01 §6 đã đính chính). Đây bật `obs_normalization` cho actor+critic
(EmpiricalNormalization của rsl_rl, lưu trong ckpt, tự áp dụng khi eval). Kỳ vọng sửa
critic, không sửa kiểu nắm.

Metric mới (reduce last): `thumb_yaw_last`, `grip_bodies_last`,
`grip_squeeze_xy_last`, `grip_net_z_last`. Cú trôi kiểu 700 sẽ hiện trên wandb.

Chưa chốt: cơ chế đối ngón. Không sửa XML. Phương án: cận dưới thumb yaw ~0.5 ở tầng
action (clamp target), hoặc nhân contact/impulse với clamp((yaw − 0.3)/0.5, 0, 1). Đo
init yaw 0.6, d 0.10 trên 35 vật: đầu ngón gần nhất 1.2 cm (hammer), 2% reset chạm;
init yaw 0.6 cần standoff ~0.12.
