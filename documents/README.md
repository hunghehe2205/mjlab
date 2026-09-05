# RobustDexGrasp → mjlab (UR5e + RH5-DG2)

Thư mục này ghi lại thiết kế, trạng thái triển khai và lịch sử thực nghiệm của
nhánh DexGrasp. Khi tài liệu mâu thuẫn, ưu tiên theo thứ tự:

1. code tại `src/mjlab/tasks/dexgrasp` và `src/mjlab/asset_zoo`;
2. [02-phase1-teacher-plan.md](02-phase1-teacher-plan.md) cho cấu hình;
3. [09-r1-plan.md](09-r1-plan.md) cho kế hoạch đang thực thi;
4. README này;
5. các báo cáo lịch sử `05–08`.

Mục tiêu cố định: teacher chỉ học grasp như paper gốc. Scripted lift là bài
test trong eval, không vào training; lift/hold/reach có nhiễu là module ngoài.

## Trạng thái hiện tại

| Hạng mục | Trạng thái |
| --- | --- |
| Robot, object assets, pre-grasp, teacher PPO | Đã triển khai |
| Single-object, enclosure gate, raw obs | Baseline hiện tại; checkpoint ổn định khoảng 62% |
| Obs normalization | Đã loại: hai smoke run không bootstrap được contact |
| Opposition gate | Đã revert; là đòn bẩy A của R1, test lại với raw obs |
| Probe reward offline, 08 §10 | Đã xong: A ↔ thumb yaw, B ↔ displacement; squeeze loại |
| R1 screening C / O / OD rồi nhánh thắng 800 iter, 09 | Đã chốt spec sau review; chưa chạy |
| Teacher 5 object | Đã cấu hình; chờ R1 qua cổng 09 §6 và smoke from-scratch 09 §7 |
| Teacher 35 object | Đã cấu hình assets/task, chưa train |
| Vision student, domain randomization, hardware | Chưa triển khai |

HEAD `999a41bb3` đã quay về cấu hình của baseline `8yelo0qc`: actor/critic dùng
raw observation, reward contact/impulse dùng enclosure gate, không có
opposition ramp hay các grip metric thử nghiệm. Hai smoke run đã kết thúc:
`hggth52d` (obs normalization + opposition) crash ở iter 201 và
`61rxeq95` (obs normalization + enclosure giống baseline) chạy đủ 300 iter;
cả hai đều giữ contact gần 0. Vì `61rxeq95` đã bỏ opposition nhưng vẫn thất
bại, bằng chứng hiện tại quy regression cho observation normalization, không
phải cho độ thưa của opposition reward.

Run `8yelo0qc` là baseline đã train dài với enclosure gate nhưng chưa có obs
normalization/opposition gate. `model_1000` đạt trung bình 62.0% và
`model_1400` đạt 60.4% trên ba seed với 38% episode vật lật. Sweep 37
checkpoint cho thấy đỉnh đơn 80.5% ở `model_1450`, nhưng tụt còn 14.1% ngay
`model_1500`; train reward không đủ để chọn checkpoint hoặc bảo đảm grip ổn
định. Ghép sweep với metric train và probe offline (08 §10) tách ra hai vùng
lỗi ngược nhau: hook do thumb không đối ngón, và push do kéo vật quá xa.
`model_1400` là điểm warm start cho R1.

Working tree có code additive chưa commit, không đổi hành vi train: eval có
`--seed`, `frac_drop`, `frac_tipped`; module `mdp/grip_metrics.py`; mode
`squeeze_xy` trong `ContactReward` không term nào dùng.

## Tài liệu chính

| File | Vai trò |
| --- | --- |
| [01-analysis.md](01-analysis.md) | Phân tích paper/repo gốc; snapshot lịch sử |
| [02-phase1-teacher-plan.md](02-phase1-teacher-plan.md) | Source of truth cho teacher hiện tại |
| [03-phase2-objects-plan.md](03-phase2-objects-plan.md) | Trạng thái pipeline object và kế hoạch 5 → 35 object |
| [04-phase3-student-plan.md](04-phase3-student-plan.md) | Thiết kế student; chưa triển khai |
| [09-r1-plan.md](09-r1-plan.md) | Kế hoạch R1 (spec, lệnh chạy, cổng, fallback) và lộ trình 5 → 35 object |

## Báo cáo lịch sử

| File | Nội dung |
| --- | --- |
| [05-audit-fixes.md](05-audit-fixes.md) | Audit port ban đầu và danh sách lỗi |
| [06-experiment-report.md](06-experiment-report.md) | Eval teacher và quét physics |
| [07-control-authority.md](07-control-authority.md) | Gravcomp, gain, standoff và reward clip |
| [08-log-audit-fixes.md](08-log-audit-fixes.md) | Chuỗi run W&B, saturation, ablation normalization, sweep 37 checkpoint và probe reward |
| [ATTEMPTS.md](ATTEMPTS.md) | Ghi chú smoke train rất sớm |

Các file lịch sử giải thích vì sao code thay đổi, nhưng thông số bên trong có
thể đã bị thay thế. Không dùng chúng làm cấu hình chạy hiện tại nếu chưa đối
chiếu với `02`.

## Quyết định hiện tại

- Giữ IK pre-grasp làm điểm xuất phát; closed-loop reach/noise là bước sau.
- Giữ raw observation cho cả actor và critic; không bật lại empirical
  normalization toàn observation.
- Teacher grasp-only; không thêm lift hay hold vào training.
- Ổn định single-object bằng R1 trước khi mở curriculum 5 object: screening
  C / O / OD tách opposition ramp khỏi displacement, mỗi nhánh chỉ đổi một
  biến so với đối chứng; obj_vel/obj_qvel của paper hoãn vì chưa có bằng
  chứng.
- Warm start reset optimizer (`load_optimizer=False`); R1 chứng minh giữ
  policy trong vùng tốt, smoke from-scratch mới chứng minh học từ đầu.
- Scripted lift success là metric chọn checkpoint; train reward chỉ dùng để
  chẩn đoán. Cổng qua phase ở 09 §6; `frac_tipped` chỉ chẩn đoán.
- Bộ 5 object đầu là vật đứng cao ≥ 8 cm; vật phẳng để batch sau (09 §8).
- Student của paper gốc không dùng tactile thật. Dùng force sensor RH5-DG2 là
  một extension riêng và phải có ablation nếu được triển khai.
