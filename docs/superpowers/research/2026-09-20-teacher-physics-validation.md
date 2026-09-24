# Kiểm chứng teacher grasp → lift

Ngày kiểm chứng: 2026-09-20.

> Cập nhật 2026-09-24: kiểm tra bổ sung phát hiện probe đẩy vật và gây
> penetration đáng kể trong pha approach, trước khi đóng ngón. Kết quả lift/hold
> bên dưới vẫn đúng theo các điều kiện đã đo, nhưng chưa đủ để kết luận toàn bộ
> rollout đạt chất lượng vật lý. Xem
> [review pre-grasp và số đo mới](2026-09-24-pregrasp-review.md).
> Cấu hình úp bàn tay thay thế và kết quả kiểm chứng mới nằm trong
> [báo cáo palm-down](2026-09-24-palm-down-pregrasp.md).
> Hiện tại pre-grasp được lấy mẫu ngẫu nhiên theo pipeline của RobustDexGrasp:
> xem [pre-grasp lấy mẫu](2026-09-24-pregrasp-sampling.md).

## Kết luận và phạm vi

Đã triển khai ba bước: chốt spec, kiểm chứng grasp/lift bằng điều khiển scripted,
và thêm teacher environment cùng test. Baseline vật lý cuối cùng đạt trên cả
MuJoCo native và Warp CPU: nâng vật trên 10 cm, giữ liên tục 3 giây.

Đây là kết quả của quỹ đạo scripted dùng để kiểm chứng vật lý. Chưa có policy
được train hội tụ. Teacher khi train điều khiển toàn bộ 24 joint; không có lệnh
nâng scripted ghi đè action của policy.

- Task: `Mjlab-Grasp-Teacher-Ur5e-Rh5dg2`.
- Một box cố định, kích thước đầy đủ 6 × 6 × 12 cm, khối lượng 80 g.
- Reset về pre-grasp cố định; success kết thúc episode.
- Transport, hạ tay và thả vật nằm ngoài teacher hiện tại.

## Những quyết định đã chốt

| Thành phần | Quy ước triển khai |
| --- | --- |
| Action | 24 giá trị, clip [-1, 1]; target = q hiện tại + action × scale |
| Scale | Arm 0.10 rad; hand 0.50 rad mỗi control step |
| Target | Tính một lần, giữ trong 10 physics substeps, clamp hard limits |
| Δq | q đo được trừ target thực sự đang giữ |
| Tần số | Physics 200 Hz, policy 20 Hz |
| Frame | Vị trí trừ env origin; quaternion wxyz; các env cùng hướng trục |
| Dữ liệu | EntityData và ContactSensor lấy state Warp đang chạy |
| Observation | 259 giá trị privileged cho cả actor và critic |
| Reward | Incentive dương; cost không âm với weight âm; displacement chỉ XY |
| Lift | Policy tự điều khiển arm và hand; reward lift có điều kiện tiếp xúc |
| Success | Cao ≥10 cm, tốc độ tịnh tiến ≤0.05 m/s, góc ≤1 rad/s, ít nhất hai ngón có lực, vật rời bàn, liên tục 3 s |
| Reset | Reset riêng từng env, gồm target, contact history và hold counter |

Các reward thành phần được RewardManager ghi riêng. Metrics bổ sung gồm độ cao,
tốc độ vật, lực tiếp xúc, tiến độ giữ và success.

## Điều chỉnh collision và lý do

Với cylinder pad gốc, cùng một quỹ đạo có thể thành công trên native nhưng
tuột trên Warp. Đổi tư thế ngón có trường hợp đảo ngược kết quả này. Vì vậy,
native pass một mình chưa đủ làm bằng chứng cho backend dùng train.

Warp 3.11.0 cảnh báo cylinder-box chỉ tạo tối đa một contact cho mỗi cặp geom.
Teacher hiện dùng một bản collision riêng: các cylinder ở DIP được thay bằng
capsule hoặc sphere nằm bên trong cylinder cũ. Giữ nguyên vị trí, hướng, visual,
joint limits, actuator gains và body mass/inertia; không sửa asset robot dùng
chung. Test đối chiếu xác nhận body mass/inertia bằng nhau và pad mới nằm trong
biên pad cũ.

Đây là xấp xỉ hình học có chủ đích để tạo baseline tiếp xúc. Kết quả cho thấy
cấu hình cuối hoạt động trên cả hai backend; chưa chứng minh giới hạn multicontact
là nguyên nhân duy nhất của mọi trường hợp tuột. Các cylinder khác vẫn còn trong
scene nên cảnh báo Warp vẫn có thể xuất hiện.

## Số đo của cấu hình cuối

Hai backend dùng cùng scene, gains, pre-grasp, quỹ đạo và giới hạn action.
Sau reset không gắn vật vào tay, không ghi pose vật, không triệt tiêu gravity.

| Chỉ số | Native CPU | Warp CPU |
| --- | ---: | ---: |
| Success | true | true |
| Độ cao tăng tại success | 0.157307 m | 0.156612 m |
| Giữ liên tục | 3.00 s | 3.00 s |
| Tốc độ vật tại success | 0.001574 m/s | 0.002351 m/s |
| Lực normal lớn nhất trên hand tại success | 6.189228 N | 7.065795 N |
| Penetration lúc reset | 5.55e-17 m | 5.96e-8 m |

Ngưỡng penetration của probe là 1e-5 m. Số đo Warp lấy từ contacts Warp thực tế;
số đo native lấy từ MuJoCo native. Lực trong bảng là maximum tại thời điểm
success, không phải tổng lực hay peak của cả episode.

Máy kiểm chứng: macOS 26.0.1, ARM64, CPU. Phiên bản: MuJoCo 3.11.0,
mujoco-warp 3.11.0, warp-lang 1.14.0, Torch 2.9.0, rsl-rl-lib 5.5.0.

## Kiểm tra phần mềm

- `make check`: ruff format/check, ty và pyright đều đạt.
- 27 test liên quan đạt trong 26.67 giây, gồm probe native và Warp, observation
  hữu hạn, bất biến theo env translation, action/target, reset từng env,
  contact Warp thực, dấu reward và hold counter liên tục/idempotent.
- PPO smoke: 2 env × 4 rollout steps, 1 epoch, 1 mini-batch, một lượt cập nhật;
  chạy được và các tham số policy hữu hạn. Smoke chạy local, không upload model.

Các test đã chạy:

```sh
uv run pytest tests/test_grasp_teacher.py tests/test_ur5e_arm.py \
  tests/test_rh5dg2_hand.py tests/test_workstation_scene.py \
  tests/test_ur5e_rh5dg2.py tests/test_ur5e_rh5dg2_view_env.py \
  tests/test_task_configs.py -q
```

## Chạy lại và bước kế tiếp

```sh
# Backend dùng train; xuất số đo để so sánh.
uv run python -m mjlab.tasks.ur5e_rh5dg2.grasp.physics_probe \
  --backend warp --device cpu --output /tmp/grasp-warp.json

# Đối chiếu native.
uv run python -m mjlab.tasks.ur5e_rh5dg2.grasp.physics_probe \
  --backend native --output /tmp/grasp-native.json
```

Trước khi train dài trên CUDA, chạy lại probe với `--device cuda:0` và test trên
máy đó. Sau đó chạy PPO baseline, theo dõi success cùng từng reward, lực, độ cao
và tốc độ. Chưa kiểm chứng CUDA, throughput nhiều env hay learning convergence.
Baseline một box và một pre-grasp không chứng minh khả năng grasp vật hoặc tư thế
khác; pad xấp xỉ cũng chưa được hiệu chuẩn theo phần cứng thật.

Spec hiện hành:
[Teacher grasp-and-lift](../specs/2026-09-16-teacher-phase-dexgrasp-design.md).
