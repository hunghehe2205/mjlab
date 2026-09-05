# 08 — Đối chiếu thông số với bản gốc + đọc log wandb, và các fix

> **Nhật ký thực nghiệm.** Các section được giữ theo thời gian; quyết định cũ
> có thể bị section sau thay thế. Cấu hình chạy hiện tại được tổng hợp ở
> [02-phase1-teacher-plan.md](02-phase1-teacher-plan.md).

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

## 6. Run 8yelo0qc (stage1 gate, 5fce3aa): bão hoà và grip drift

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
01 §6 đã đính chính). Run này dùng raw observation. Đề xuất bật normalization và
thêm grip metric sau đó là giả thuyết thử nghiệm, không phải cấu hình của run.

Commit `20945e120` từng thêm normalization cùng grip metrics; `9a62eaacb` thêm
`OppositionGatedContact` sau enclosure gate. Cả hai thay đổi đã bị revert. HEAD
`999a41bb3` không chứa normalization, opposition hay grip metrics.

## 7. Run hggth52d (obs normalization + opposition, 9a62eaacb)

Smoke run single-object: 1024 env, seed 42. Run crash ở iter 201; contact giữ
gần 0 toàn bộ run. Value loss giảm `156 → 1.51` không phải bằng chứng critic
tốt hơn: vì policy không tìm thấy contact, return gần hằng và dễ fit.

Không thể quy thất bại cho opposition từ run này vì normalization và reward
được đổi cùng lúc. Cần ablation chỉ thay một biến.

## 8. Run 61rxeq95: ablation cô lập observation normalization

Run này giữ enclosure reward giống `8yelo0qc`, bỏ opposition ramp, nhưng vẫn
bật normalization cho actor và critic. Nó chạy đủ 300 iteration mà contact
gần như không xuất hiện:

| Run | Reward | Obs norm | contact iter 100–200 | contact iter 200–300 |
| --- | --- | --- | ---: | ---: |
| `8yelo0qc` | enclosure | không | 1.98 | 2.95 |
| `61rxeq95` | enclosure | có | 0.000 | 0.001 |
| `hggth52d` | enclosure × opposition | có | 0.000 | crash |

Biến chung duy nhất của hai smoke run thất bại là empirical observation
normalization. Với các chiều contact gần hằng ở đầu training, std chạy gần 0;
khi flag hiếm đầu tiên đổi sang 1, normalized value trở thành outlier rất lớn.
af-vector và impulse cũng bị đổi thang mạnh, trong khi thống kê đầu vào actor
tiếp tục trôi. Commit `6ea6c7a03` tắt normalization là quyết định đúng; commit
`999a41bb3` hoàn tất revert source về baseline.

Opposition ramp chưa từng được test sạch với raw obs, nên chưa có bằng chứng
kết luận nó tốt hay xấu. Nếu cần cải thiện critic, critic-only normalization
hoặc fixed scaling phải là run riêng và không được đổi reward đồng thời.

## 9. Sweep eval 37 checkpoint của 8yelo0qc

Potted meat can, scripted lift; `reachable=100%` và `track_err≈0` ở mọi
checkpoint. Vì vậy khác biệt đến từ chất lượng grip, không phải IK hay tracking
của arm.

| iter | success | gain (cm) | iter | success | gain (cm) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0% | −0.0 | 950 | 32.0% | +2.7 |
| 50 | 0.0% | −0.0 | 1000 | 59.4% | +7.9 |
| 100 | 1.6% | +1.0 | 1050 | 53.9% | +7.0 |
| 150 | 24.2% | +2.4 | 1100 | 39.8% | +4.4 |
| 200 | 33.6% | +3.9 | 1150 | 59.4% | +7.7 |
| 250 | 43.0% | +5.6 | 1200 | 55.5% | +6.8 |
| 300 | 45.3% | +6.0 | 1250 | 10.9% | −0.3 |
| 350 | 28.9% | +3.1 | 1300 | 55.5% | +6.9 |
| 400 | 39.8% | +5.0 | 1350 | 64.1% | +8.1 |
| 450 | 37.5% | +4.5 | 1400 | 64.1% | +8.2 |
| 500 | 14.8% | +0.8 | 1450 | 80.5% | +10.9 |
| 550 | 32.8% | +3.6 | 1500 | 14.1% | −0.2 |
| 600 | 38.3% | +4.1 | 1550 | 1.6% | −2.1 |
| 650 | 10.9% | −0.4 | 1600 | 0.8% | −1.9 |
| 700 | 0.0% | −1.6 | 1650 | 8.6% | −0.9 |
| 750 | 0.0% | −1.6 | 1700 | 13.3% | +0.2 |
| 800 | 7.0% | −0.7 | 1750 | 63.3% | +7.8 |
| 850 | 7.0% | −1.1 | 1800 | 47.7% | +4.7 |
| 900 | 5.5% | −0.8 |  |  |  |

Đỉnh đơn 80.5% ở iter 1450 không bền: chỉ 50 iteration sau còn 14.1%. Cùng
với train reward/contact trơn, sweep củng cố kết luận reward tĩnh có plateau
giữa hook và wrap grasp. Checkpoint phải được chọn bằng scripted lift eval;
hướng cải thiện tiếp theo phải tạo tín hiệu phân biệt chất lượng grip dưới
lift, không tune PPO theo train reward.

## 10. Ghép sweep với metric train, model_1400 ba seed, probe reward

### Ghép sweep với metric train

Metric train lấy trung bình 10 iteration trước mỗi checkpoint; reward/bước =
`Episode_Reward × 0.2`; dịch vật = trung bình theo thời gian trong episode.

| Nhóm theo eval | n | reward/ep | contact/bước | dịch vật |
| --- | ---: | ---: | ---: | ---: |
| tốt, ≥ 50% | 9 | 74.9 | 0.81 | 6.4 cm |
| trung bình, 15–50% | 12 | 55.3 | 0.65 | 4.6 cm |
| xấu, < 15% | 13 | 65.8 | 0.75 | 5.6 cm |

Nhóm xấu có reward cao hơn nhóm trung bình. Spearman giữa success và metric
train đều yếu: reward +0.34, contact +0.21, impulse +0.37, displacement −0.25,
value loss −0.42. Hai vùng sụp có dấu hiệu train ngược nhau:

| Vùng | success | gain | reward | dịch vật | arm action |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: 650–900 | 0–11% | −0.4 đến −1.6 cm | 53 → 65 | 3.2–3.5 cm | 1.2–1.3 |
| tốt: 1300–1450 | 55–80% | +7 đến +11 cm | 78 | 5.4–7.4 cm | 1.6–1.7 |
| B: 1500–1700 | 1–14% | −0.9 đến −2.1 cm | 73–74 | 8–8.9 cm | 1.7–1.9 |

Gain âm nghĩa là vật kết thúc thấp hơn lúc đầu, tức bị lật khi nâng. Value
loss nhảy tại 1250 (115 so với 46–90) và 1500–1650 (128–173 so với 55–106),
là chỉ báo "policy vừa đổi kiểu", không phải tốt hay xấu.

Cổng "≥ 60% ở ba checkpoint liên tiếp" đã đạt tại 1350 / 1400 / 1450 và sụp
ngay 1500, nên cổng phải chặt hơn (09 §6).

### model_1400, ba seed

128 episode mỗi seed: 56.2 / 64.1 / 60.9%, mean 60.4%, gain +7.6 cm,
`frac_tipped` 38.3%. Chọn làm warm start cho R1.

### Probe 8 đại lượng × 4 vùng

potted_meat_can, 10 bước cuối pha grasp, qua `mdp/grip_metrics.py`. h_load =
`Σ w|f_xy|`, net = `|Σ w f_xy|`, squeeze = h_load − net, balance =
squeeze / h_load. Kết quả lưu trên server eval `eval_results/8yelo0qc/probe_grip.json`.

| Vùng | success | h_load | net | squeeze | balance | thumb_yaw | n_contact | disp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tốt | 67% | 4.66 | 2.94 | 1.72 | 0.369 | 0.880 | 7.71 | 9.8 cm |
| trung bình | 41% | 3.83 | 2.33 | 1.50 | 0.397 | 0.756 | 5.62 | 7.4 cm |
| A, hook | 3.5% | 1.75 | 1.10 | 0.65 | 0.365 | 0.266 | 6.06 | 5.1 cm |
| B, push | 3.6% | 4.09 | 2.59 | 1.50 | 0.370 | 0.819 | 8.33 | 17.6 cm |

Spearman với success: squeeze +0.74, net +0.70, thumb_yaw +0.67, h_load +0.63,
balance +0.31, tilt +0.28, n_contact −0.01, disp −0.12.

Kết luận:

- thumb_yaw cô lập A: 0.27 ≈ init 0.3 so với 0.76–0.88 ở ba vùng còn lại.
- disp cô lập B: 17.6 cm so với 9.8 cm ở vùng tốt; mọi đại lượng lực chỉ
  chênh khoảng 15% giữa tốt và B.
- balance phẳng 0.37 ở cả bốn vùng: squeeze chỉ là 0.37 lần h_load, không
  mang thông tin mới. Giả thuyết "internal squeeze phân biệt grip" bị bác.
- net tăng theo success, nên không phạt net-force.
- tilt trong pha grasp nhiễu theo frame vật, không dùng làm tín hiệu.

Quyết định rút ra và spec R1 ở [09](09-r1-plan.md): ramp thumb yaw cho A,
hệ số object stability gốc cho B, không squeeze, không hinge, không lift
trong train.
