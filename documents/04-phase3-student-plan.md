# Student — single-view, history và distillation

Toàn bộ phase này hiện là kế hoạch; chưa có student environment, runner,
checkpoint hoặc kết quả. Điều kiện vào là teacher ổn định trên multi-object,
không chỉ build được task 35 object. Teacher lẫn student đều grasp-only như
paper; lift, hold và reach có nhiễu là module ngoài, không vào training.

## Phạm vi paper gốc

Student RobustDexGrasp nhận single-view point cloud và lịch sử proprioception,
dùng LSTM để tái dựng contact privileged rồi học bằng mixed
imitation/reinforcement curriculum. Kết quả real-world 94.6% thuộc pipeline
đầy đủ này cùng domain randomization và hardware stack.

Paper gốc không dựa vào tactile sensor thật. RH5-DG2 có các pad/force-sensor
body trong model, nhưng đưa tín hiệu này vào student là một extension của port,
không phải reproduction nguyên bản. Nếu triển khai, phải báo riêng:

- baseline full contact reconstruction như paper;
- sensor-assisted student;
- ablation để định lượng lợi ích của sensor.

## A. Observation student

- [ ] Single-view point cloud lấy một lần ở reset bằng camera/raycast.
- [ ] Affordance vector từ keypoint hiện tại tới cloud tĩnh.
- [ ] Proprioception history 10 bước: joint position và PD error.
- [ ] Observation lag 1–2 control step như paper.
- [ ] Tách rõ observation dùng được trên robot thật và privileged critic input.
- [ ] Tùy chọn sensor-assisted 6 kênh, không trộn vào baseline paper.

Teacher hiện có stochastic target delay ở physics substep đầu. Cơ chế đó
không thay cho observation lag 1–2 control step của student.

## B. Kiến trúc và thuật toán

- [ ] LSTM state-history encoder.
- [ ] Head tái dựng contact flags và impulses/latent tương ứng.
- [ ] Wrapper `RslRlDistillationCfg` cho mjlab hoặc runner riêng.
- [ ] Khởi tạo policy từ teacher tương thích.
- [ ] Mixed curriculum:
  `ppo_ratio = min(iteration * 5e-4, 1.0)`.
- [ ] Loss logging tách imitation, reconstruction, policy và value.

`rsl_rl` có `DistillationRunner`, nhưng cần kiểm tra nó có đáp ứng mixed
DAgger+PPO và reconstruction loss của paper hay không trước khi chọn.

## C. Domain randomization

- [ ] Friction object/finger.
- [ ] PD gain arm/hand.
- [ ] Joint-position và frame calibration noise.
- [ ] Action/observation delay.
- [ ] Perception bias/point-cloud perturbation.
- [ ] Object mass, pose và camera extrinsics phù hợp hardware.
- [ ] External-force robustness evaluation.

Các range cũ trong lịch sử chỉ là điểm xuất phát; phải đo lại cho UR5e +
RH5-DG2, không sao chép thẳng từ Allegro/RaiSim.

## D. Đánh giá

- [ ] Teacher và student chạy cùng object/seed/reset.
- [ ] Scripted lift success per object.
- [ ] Ablation từng nguồn noise và từng nguồn observation.
- [ ] Generalization sang object chưa train.
- [ ] Latency và closed-loop rate trên hardware target.

## Ngoài phạm vi hiện tại

ROS/hardware interface, camera calibration, safety limits và real-world
deployment cần một plan riêng. Không dùng con số real-world của paper để mô tả
trạng thái repo trước khi các phần này tồn tại và được đánh giá.
