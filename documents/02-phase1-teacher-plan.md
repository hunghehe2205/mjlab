# Teacher hiện tại — privileged PPO

Tài liệu này là source of truth cho teacher trên UR5e + RH5-DG2. Nó mô tả
code tại commit `999a41bb3` cộng phần additive trong working tree (eval seed,
`grip_metrics.py`, mode `squeeze_xy` chưa dùng), và tách rõ ba mức: đã triển
khai, đã kiểm chứng qua eval, và đang thử nghiệm. Kế hoạch đang thực thi và
thay đổi reward sắp tới nằm ở [09](09-r1-plan.md).

## Mục tiêu và cổng chuyển phase

Teacher chỉ học grasp như paper gốc; lift chỉ là bài test trong eval. Thứ tự:

1. single-object qua cổng 09 §6: 5 checkpoint liên tiếp mean ba seed ≥ 60%,
   mean của 5 checkpoint ≥ 70%, không cú tụt liền kề quá 15 điểm, thumb yaw
   giữ ≥ 0.6, displacement không tăng quá 2 cm trong cửa sổ;
2. giữ nguyên cấu hình đã qua cổng để train curriculum 5 object;
3. chỉ mở toàn bộ 35 object sau khi 5 object ổn định.

Run `8yelo0qc` đã chứng minh enclosure-gated reward có thể đạt khoảng 62%,
nhưng checkpoint trôi mạnh và reward train không dự báo được lift success.
Hai smoke run observation normalization đã thất bại; HEAD đã quay lại raw obs
và enclosure gate của baseline.

## Trạng thái run hiện tại

| Run | Reward | Obs norm | Kết quả |
| --- | --- | --- | --- |
| `8yelo0qc` | enclosure | không | contact 1.98 ở iter 100–200, 2.95 ở 200–300 |
| `61rxeq95` | enclosure, giống baseline | có | contact 0.000 rồi 0.001; kết thúc 300 iter |
| `hggth52d` | enclosure × opposition | có | contact 0; crash ở iter 201 |

`61rxeq95` là ablation quyết định: sau khi bỏ opposition, reward giống
`8yelo0qc` nhưng policy vẫn không bootstrap. Biến chung của hai run thất bại
là empirical observation normalization. Vì vậy không còn kết luận “reward
quá thưa” từ hai smoke run này; opposition ramp chưa từng được test sạch với
raw observation.

Value loss thấp trong `hggth52d` không chứng minh critic tốt hơn: khi contact
không xuất hiện, return gần như hằng và bài toán value trở nên dễ giả tạo.

## Assets và scene

- [x] UR5e + RH5-DG2, 24 khớp điều khiển, gravcomp và gain đã hiệu chỉnh.
- [x] Site `grasp_center`, collision cho pad, init finger pose thumb yaw 0.3.
- [x] 35 object từ cohort RobustDexGrasp cùng metadata 200 affordance points,
  centroid và lowest point.
- [x] Object collision hiện dùng một convex hull của
  `top_watertight_tiny.obj`.
- [x] `VariantEntityCfg` cho curriculum 5 object và full 35 object.
- [ ] Chưa ghép mesh bottom/non-affordance và chưa chạy drop-test tự động cho
  toàn bộ cohort.

Ba task được đăng ký:

| Task | Object |
| --- | --- |
| `Mjlab-DexGrasp-UR5eRH5DG2-Single` | potted meat can |
| `Mjlab-DexGrasp-UR5eRH5DG2-Phase1` | 5 object |
| `Mjlab-DexGrasp-UR5eRH5DG2` | 35 object |

## Simulation và reset

- Physics timestep: `0.005 s`.
- Action decimation: `40`, tương đương policy 5 Hz.
- MuJoCo: Newton, implicit-fast, pyramidal cone, `impratio=1`.
- Episode: 14 s, 70 action.
- Default full task: 88 env theo trọng số object của repo gốc.
- Reset pose object: vùng cực gốc, sampling edge-biased được bật khi train.
- Camera cố định: `[0.035, -0.58, 1.531]`.
- IK: analytic UR5e, chọn pre-grasp collision-free.
- Standoff theo site `grasp_center`: `0.10 m`, không phải 0.25 m của Allegro.

IK pre-grasp vẫn là điểm xuất phát chuẩn. Nhiễu pose để tạo closed-loop reach
robust là bước sau khi grasp single/multi-object đã ổn định; chưa trộn vào
smoke run hiện tại.

## Action

- Relative joint-position target, arm scale `0.005`, finger scale `0.015`.
- Absolute target được clamp về joint limits.
- Với xác suất 0.5, physics substep đầu của mỗi control step dùng target trước.
  Đây là delay đang có trong teacher; không phải observation lag 1–2 control
  step của student trong paper.

## Observation

Actor và critic dùng cùng privileged observation 191 chiều:

| Thành phần | Dim |
| --- | ---: |
| joint position + PD error | 48 |
| contact flags + impulse | 32 |
| hand keypoint height + arm link height | 30 |
| grasp center + wrist orientation/difference | 9 |
| affordance vectors | 72 |
| Tổng | 191 |

`obs_normalization=False` cho cả actor và critic, khớp repo gốc và baseline
`8yelo0qc`. Không bật lại `EmpiricalNormalization` cho toàn observation:
contact flag gần hằng trong exploration trở thành outlier rất lớn khi lần đầu
chuyển từ 0 sang 1, còn af-vector bị khuếch đại và phân phối đầu vào actor trôi
liên tục. Nếu cần giúp critic, test critic-only normalization hoặc fixed
per-feature scaling như một ablation riêng.

## Reward hiện tại

| Term | Weight | Ghi chú |
| --- | ---: | --- |
| affordance distance | +0.5 | weighted nearest-point distance |
| affordance contact | +1.5 | enclosure gate |
| affordance impulse | +0.5 | enclosure gate |
| table log-barrier/contact/impulse | −0.03/−1/−0.5 | |
| arm height/contact/impulse/collision | −0.05/−0.1/−0.1/−1 | |
| object vel/angular vel/displacement | −5/−0.1/−2 | mềm hơn paper; nhánh OD thử displacement −5, vel/qvel giữ nguyên |
| wrist vel/angular vel | −1/−0.1 | |
| arm joint velocity | −1 | |

Không clip reward tổng và không có terminal penalty, khớp hành vi thực tế của
code gốc. `ContactReward` có thêm mode `squeeze_xy` trong working tree; probe
08 §10 cho thấy nó không phân biệt được grip nên không term nào dùng.

### Enclosure gate

HEAD dùng `EnclosureGatedContact`. Pipeline chính xác:

```text
contact/impulse thô
  → gate nhị phân: thumb tip + ít nhất 2 non-thumb fingertips
```

Không có bằng chứng gate cliff gây các cú tụt eval của `8yelo0qc`. Cơ chế đã
đo được là reward contact/impulse bão hòa và không phân biệt hook grasp với
wrap grasp. `OppositionGatedContact` từng được thêm ở `9a62eaacb` rồi revert;
do run đó đồng thời bật obs normalization, chưa thể kết luận opposition giúp
hay hại. Probe 08 §10 sau đó cho thấy thumb yaw là đại lượng duy nhất cô lập
vùng hook, nên R1 khôi phục ramp này với raw obs (09 §1, §3).

## Metrics và eval

HEAD không còn các metric thử nghiệm `thumb_yaw_last`, `grip_bodies_last`,
`grip_squeeze_xy_last` và `grip_net_z_last`. Các số phân tích hook/wrap trong
§08 đến từ probe offline qua `mdp/grip_metrics.py`. R1 thêm lại metric online
`thumb_yaw_last`, `object_displacement_last`, `h_load_last`, `push_last`
(09 §3).

Eval chạy 70 bước policy rồi 90 bước scripted lift, target nâng 0.15 m. Success
khi độ cao cuối của object tăng hơn 0.10 m. Working tree đã có `--seed` và
JSON thêm `frac_drop` (gain dưới −1 cm) và `frac_tipped` (trục z object lệch
quá 45°). Protocol chuẩn: 128 episode × seed 0/1/2, báo mean và min theo seed.

## PPO

- Actor/critic: MLP 128 × 128, LeakyReLU.
- Gaussian scalar std, init 1.0, floor 0.2.
- Learning rate cố định `5e-4`.
- Gamma `0.996`, lambda `0.95`.
- PPO clip `0.2`, value loss coeff `0.5`, entropy coeff 0.
- 4 epoch, 4 minibatch, max grad norm `0.5`.
- Một rollout chứa đủ 70 bước của episode.

Log hiện tại không ủng hộ việc tune PPO trước. PPO train trơn trong
`8yelo0qc`; vấn đề là reward tĩnh không xếp hạng đúng chất lượng grip dưới
scripted lift.

## Việc tiếp theo

Toàn bộ ở [09](09-r1-plan.md). Tóm tắt:

1. Giữ actor/critic raw obs; không tiếp tục hai smoke run normalization.
2. Screening ba nhánh từ `model_1400`, optimizer mới (`load_optimizer=False`),
   LR 1e-4, 300 iter: C đối chứng, O chỉ opposition ramp, OD ramp +
   displacement −5. obj_vel/obj_qvel giữ −5/−0.1; hệ số paper −15/−0.2 hoãn
   vì probe không có bằng chứng (09 §1–§2).
3. Nhánh thắng chạy 800 iter. Eval mỗi 50 iteration, 3 seed, chọn checkpoint
   theo scripted lift; cổng 09 §6; `frac_tipped` chỉ chẩn đoán.
4. Không thêm lift hay hold vào training. Fallback theo chữ ký đo được, vẫn
   grasp-only (09 §6).
5. Smoke from-scratch 300 iter cho cấu hình thắng (09 §7), rồi mới mở 5
   object; noise pre-grasp/reach và 35 object để sau.
6. Bỏ mode `squeeze_xy` khỏi `ContactReward` trước khi commit; giữ helper
   trong `grip_metrics.py`.
