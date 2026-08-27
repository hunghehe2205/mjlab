# Báo cáo thực nghiệm — eval teacher + quét physics

Chạy sau khi 3 fix của [05-audit-fixes.md](05-audit-fixes.md) được áp (`ac33c3b`).
Run tham chiếu: wandb `mjlab-dexgrasp/vcb6edm2`, checkpoint `model_2400.pt`.

## Tóm tắt

1. Ba fix **hoạt động đúng** — contact signal sống lại, metric trung thực.
2. Eval thật: **0.0% lift success** trên 6 vật dễ nhất.
3. **Bug mới, nghiêm trọng nhất: ratchet trọng lực.** Với `action = 0` (nghĩa là
   "giữ nguyên"), arm tự sụt **0.31 m** và ngón tay tự cuộn tới **0.97 rad** mỗi
   episode. Policy chưa bao giờ thực sự điều khiển được robot.
4. Giả thuyết physics ở [05 §3](05-audit-fixes.md) **sai** — không phải solver
   iterations. Là **timestep**.
5. Cấu hình đã kiểm chứng: `gravcomp` + `dt=5 ms` + `cone=pyramidal` →
   jitter **12×** thấp hơn, arm sụt **về 0**, và **nhanh hơn 2.4×** hiện tại.

---

## 1. Ba fix hoạt động

Đo lại trên env (70 bước × 8 env, ngón siết vào `potted_meat_can`):

| | trước | sau |
|---|---|---|
| contact-body kích hoạt | 189 | **552** |
| impulse max (clip 0.1/0.2) | 1.6657 (bão hoà 16×) | **0.2478** (trong dải) |
| `affordance_contact`/bước (max 1.5) | ~0.003 | **0.120** |
| obs dim | 191 | 191 (pad gộp, không nở obs) |

Trên run thật, cùng iteration (~2300–2400):

| | `advid9la` | `vcb6edm2` | |
|---|---|---|---|
| `affordance_contact`/bước | 0.0033 | **0.0400** | ×12 |
| `affordance_impulse`/bước | 0.0044 | **0.0111** | ×2.5 |
| `object_lift_height_max` | 0.118 m (giả) | **0.013 m** (thật) | |
| `lift_success` | 0.373 (giả) | 0.0036 (thật) | |
| learning rate | 1e-5 (sàn) | 3.8e-4 | |

`PAD_PARENT_INDICES = (3, 6, 9, 12, 15, 0)` đúng; cộng vector trước rồi mới lấy
norm — khớp bản gốc.

## 2. Eval: 0.0%, và **hai** nguyên nhân độc lập

`dexgrasp-eval` trên `model_2400.pt`, 64 env/vật, uniform sampling:

```
potted_meat_can / tomato_soup_can / tuna_fish_can /
sugar_box / pudding_box / master_chef_can  ->  0.0% lift success
```

Chẩn đoán sâu (`potted_meat_can`, 64 env):

```
[cuối pha grasp] số contact body chạm vật: mean=0.00  max=0  | %env có ≥2 contact: 0%
[IK lift]        reachable 64/64, cần di chuyển 0.397 rad, 0.00441 rad/bước
                 trần action 0.005 rad/bước  ->  ĐỦ
[sau lift]       sai số bám khớp cuối: 0.577 rad  (lớn hơn cả 0.397 rad cần đi)
[sau lift]       grasp_center nâng được: -0.179 m   (mục tiêu +0.15)
[sau lift]       object nâng: mean 0.003 m, max 0.017 m, >0.10 m: 0/64
```

Hai kết luận tách biệt:

- **Policy không cầm được gì** — 0/64 env có bất kỳ contact nào ở bước 70.
- **Script lift cũng hỏng** — arm đi sai hướng, grasp_center **sụt** 0.18 m thay
  vì nâng 0.15 m, và sai số cuối lớn hơn quãng đường cần đi.

Cả hai quy về cùng một gốc ở §3.

## 3. Bug mới: ratchet trọng lực

### Hiện tượng

`action = 0` với `RelativeJointPositionAction` nghĩa là `target = qpos hiện tại`
— tức "giữ nguyên vị trí". Đo thực tế, 70 bước, robot không nhận lệnh gì:

| bước | trôi khớp arm (max) | grasp_center z | sụt |
|---|---|---|---|
| 5 | 0.041 rad | 1.048 | −0.034 |
| 20 | 0.171 | 0.949 | −0.132 |
| 50 | 0.354 | 0.814 | −0.268 |
| **70** | **0.391** | **0.802** | **−0.280** |

Từng khớp sau 70 bước: `shoulder_lift −0.334`, `elbow −0.215`, `wrist_1 +0.118`.

Ngón tay còn tệ hơn:

| bước | trôi ngón (max) | trung bình 18 khớp |
|---|---|---|
| 10 | 0.144 rad | 0.044 |
| 35 | 0.367 | 0.106 |
| **70** | **0.968** (`R_pinky_dip`) | **0.369** |

### Cơ chế

Action là `target = qpos_hiện_tại + delta`. Với `delta = 0`, PD ổn định ở
`qpos − τ/kp` (lún đúng bằng sai số tĩnh). Bước điều khiển kế tiếp **neo lại
target vào vị trí đã lún**. Không có gì sửa → tích luỹ tuyến tính.

```
khớp            mô-men trọng lực   kp     sai số tĩnh τ/kp    trần action/bước
shoulder_lift      11.6 N·m       2000      0.0058 rad            0.005
elbow              17.4 N·m       2000      0.0087 rad            0.005   <-- lún > trần
```

**Độ lún mỗi bước lớn hơn action tối đa.** Policy không thể giữ arm đứng yên,
chứ đừng nói nâng vật. Với ngón (`kp = 1.0`) tỉ lệ còn tệ hơn nhiều.

Không phải bão hoà mô-men: giới hạn ±150 N·m, chỉ cần 17.4.

### Vì sao bản gốc không bị

Bản gốc dùng cùng sơ đồ (`actionMean_r_ = gc_r_`), nhưng gain **đã nhận dạng từ
phần cứng**: UR5 kp ≈ **16000** (`UR5Identification_id5hz.txt`), Allegro kp =
**600** (`Allegrotemp.txt`). Với kp = 16000, lún = 17.4/16000 = **0.0011
rad/bước** — dưới trần action 5×. Port dùng gain menagerie (kp 2000/500) và
gain ngón bịa (kp 1.0). Đây chính là **D1 trong [05 §5](05-audit-fixes.md)**,
nhưng nó không phải rủi ro sim2real hạng 7 — nó là **blocker hạng 1**.

### Nâng kp **không** phải lời giải

| cấu hình | kết quả @ dt=10 ms |
|---|---|
| kp=2000/kd=500 (menagerie) | trôi 0.410 rad, sụt −0.292 m |
| kp=8000/kd=400 | **mất ổn định** (trôi 3.01 rad) |
| kp=16000/kd=500 | **mất ổn định** (3.02 rad) |
| kp=16000/kd=800 | **mất ổn định** (2.96 rad) |

MuJoCo `implicitfast` xử lý damping ngầm nhưng **stiffness tường minh**, nên
ổn định cần `kp·dt²/M ≲ 1`. RaiSim giải PD như ràng buộc nên chịu được kp 16000.
Không port thẳng con số được.

### Lời giải đã kiểm chứng: `gravcomp`

| cấu hình | trôi arm | grasp_center sụt | \|ω\| object |
|---|---|---|---|
| baseline dt=10 ms | 0.393 rad | −0.305 m | 1.225 |
| **gravcomp dt=10 ms** | **0.0000** | **+0.0000** | 1.011 |
| dt=5 ms | 0.403 | −0.249 | 0.495 |
| **gravcomp + dt=5 ms** | **0.0000** | **+0.0000** | **0.100** |
| gravcomp *chỉ arm* dt=10 ms | 0.100 | −0.093 | 0.894 |

Phải bù cho **toàn bộ robot**, không chỉ arm: bàn tay 0.574 kg treo ở flange,
nếu không bù thì vẫn còn 0.093 m sụt.

**Với sim2real đây là mô hình đúng hơn, không phải gian lận.** Bộ điều khiển
UR5e thật chạy vòng servo có bù trọng lực dựa trên mô hình động lực học, và ta
khai báo khối lượng/CoG của tool (bàn tay) trong controller. `gravcomp=1` chính
là mô hình hoá điều đó. Lưu ý còn lại: ngón tay thật có thể không được bù —
xem §6.

## 4. Quét solver: giả thuyết ở 05 §3 sai

Thiết kế **ghép cặp** (seed 12345, cùng pose object cho mọi cấu hình), 6 env,
50 bước, 3 vật, robot đứng im. Chỉ số: `|ω|` trung bình của object.

| cấu hình | \|ω\| tb | so baseline | chi phí |
|---|---|---|---|
| **dt5ms + pyramidal** | **0.830** | **0.40×** | **0.41×** |
| dt5ms | 0.863 | 0.42× | 1.78× |
| i100 + dt5ms | 0.876 | 0.43× | 1.87× |
| pyramidal | 1.764 | 0.86× | 0.21× |
| i100 + imp1 + pyr | 1.978 | 0.96× | 0.20× |
| iterations=100 | 2.029 | 0.99× | 1.04× |
| impratio=1 | 2.052 | 1.00× | 0.82× |
| baseline | 2.058 | 1.00× | 1.00× |

**Đính chính [05 §3](05-audit-fixes.md).** Ở đó tôi nghi ma sát chưa hội tụ
(`iterations=10` + `impratio=10` + `elliptic`). Đo được:

- `iterations` 10 → 100: **0.99×**, không tác dụng.
- `impratio` 10 → 1: **1.00×**, không tác dụng.
- **`timestep` 10 → 5 ms: 0.42×** — đây mới là nguyên nhân.

Giải thích: `solref = "0.02 1"` mặc định → hằng số thời gian contact 20 ms. Ở
dt = 10 ms chỉ có **2 substep** trong một hằng số thời gian — sát ngưỡng tối
thiểu. Ở dt = 5 ms có 4 → ổn định.

Phần thưởng kèm theo: `cone="pyramidal"` **rẻ hơn 5×** và còn tốt hơn một chút.
Kết hợp với dt=5 ms cho **chi phí 0.41× so với hiện tại** — nhanh hơn 2.4× *và*
ít jitter hơn 2.4×.

## 5. Cấu hình đề xuất

`dexgrasp_env_cfg.py`:

```python
SIM_TIMESTEP = 0.005  # 0.010 -> 0.005
DECIMATION = 40  # 20 -> 40, control_dt van 0.2 s, van 70 buoc/episode
...
mujoco = (
  MujocoCfg(
    timestep=SIM_TIMESTEP,
    iterations=10,  # khong anh huong; giu hoac tra ve mac dinh 100 (+4%)
    ls_iterations=20,
    impratio=1.0,  # 10 -> 1 (mac dinh MuJoCo, khong anh huong jitter)
    cone="pyramidal",  # elliptic -> pyramidal: re hon 5x
  ),
)
```

`ur5e_rh5dg2_constants.py::get_spec()`:

```python
  arm.attach(child=hand, prefix=HAND_PREFIX, frame=frame)

  # Model the UR controller's payload-aware gravity compensation. Without it the
  # relative-position action ratchets the arm down by tau/kp every control step
  # (measured: 0.31 m of grasp-center sag per episode at zero action).
  for body in arm.bodies:
    if body.name:
      body.gravcomp = 1.0
```

> `DECIMATION` đổi 20 → 40 làm `history_length` của contact sensor đổi theo.
> Vì fix A1 dùng `mean(dim=2)` chứ không phải `sum`, thang impulse **không đổi**
> — ngưỡng 0.01 / 0.1 / 0.2 vẫn nguyên hiệu lực. Nếu chưa sửa A1 thì đổi
> decimation sẽ âm thầm rescale cả obs lẫn reward.

## 6. Đính chính các kết luận trước

- **Video iteration 2402 không phải bằng chứng policy học được.** Tôi đã đọc
  "tay tiếp cận và ôm vật" là hành vi học được. Phần lớn chuyển động đó là
  **trôi thụ động**: arm sụt 0.28 m và ngón tự cuộn 0.37 rad trung bình dù
  không có lệnh nào. Eval xác nhận: 0/64 env có contact ở bước 70.
- **Phân rã reward ở [05 §0](05-audit-fixes.md) vẫn đúng** (`object_angular_velocity`
  chiếm 53% phạt, reward thô −8.5/bước bị clip xuống −2.0), nhưng nguyên nhân
  sâu hơn: một phần jitter là do tay/arm rơi vào vật.
- **D1 (PD gain bịa) leo từ hạng 7 lên hạng 1**, và cách sửa **không** phải nâng
  kp — đã test, mất ổn định ở mọi mức.

## 7. Hướng tiếp theo

```
1. Áp cấu hình §5 (gravcomp + dt5ms + pyramidal + impratio 1)
   -> chạy lại probe idle: kỳ vọng |w| ~0.1, arm sụt 0.000
   -> đo lại FPS: kỳ vọng nhanh hơn ~2.4x

2. Tune lại gain ngón (kp, armature) ở dt=5 ms
   Ràng buộc: quán tính ngón 2.5e-6..4e-4 -> kp cao mất ổn định.
   Đòn bẩy đúng là ARMATURE (quán tính rotor sau hộp số — có thật ở tay
   thật). Quét (kp, armature) đo: (a) trôi khi action=0, (b) ổn định,
   (c) lực siết đạt được. Mục tiêu: trôi/bước << 0.015 (trần action ngón).

3. Đo lại phân bố reward thô -> quyết định reward_clip_min
   Ước tính sau khi hết jitter: thô ~ -2.1/bước, vẫn sát sàn -2.0 vì riêng
   affordance_distance ở standoff đã -2.1. Nhiều khả năng phải nới (-5)
   hoặc bỏ như bản gốc thực tế đang chạy.

4. Sửa evaluate.py: dùng z cuối thay vì peak z (bản gốc dùng z cuối);
   thêm chẩn đoán reachable/tracking-error vào output.

5. Chặng 0 smoke: 1 object, 176 env, 300 iteration.
   Cổng: affordance_contact tăng đơn điệu, object_angular_velocity KHÔNG
   còn là term lớn nhất, lr không rơi sàn.

6. Chặng 1 (5 object, 440 env) -> chặng 2 (35 object) theo
   documents/05-audit-fixes.md §7 và quy trình object đã thống nhất.
```

Bước 1–2 là điều kiện cần; chưa xong thì mọi giờ GPU đều lãng phí.

## Phụ lục — nguồn số liệu

| Số liệu | Cách đo |
|---|---|
| eval 0.0% × 6 vật | `dexgrasp-eval`, `model_2400.pt`, 64 env, uniform |
| 0/64 env có contact | `sensor_impulse` cuối bước 70, ngưỡng 0.01 |
| lift −0.179 m | FK `grasp_center` trước/sau 90 bước lift script |
| trôi arm/ngón khi action=0 | 4 env, 70 control step, `joint_pos` đầu vs cuối |
| τ trọng lực | `d.qfrc_bias[:6]` tại tư thế pre-grasp, qvel=0 |
| bảng kp | `BuiltinPositionActuatorCfg` thay `XmlActuatorCfg`, 4 env × 70 bước |
| gravcomp | `body.gravcomp = 1.0` trong `get_spec()`, 4 env × 70 bước |
| quét solver | ghép cặp seed 12345, 6 env × 50 bước × 3 vật |
