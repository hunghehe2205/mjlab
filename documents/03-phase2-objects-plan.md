# Phase 2 — Scale bộ object + curriculum

Điều kiện vào: teacher Phase 1 đạt success ổn định trên 3–5 object.
Mục tiêu: train trên cả bộ ~35 object của repo gốc, tiệm cận setup paper.

## A. Pipeline object

- [ ] Script batch convert `rsc/new_training_set` (~35 object, mesh
  `top_watertight_tiny.obj`) → MJCF: coacd convex decomposition, freejoint,
  material `object`, scale kiểm tra đơn vị.
- [ ] Batch precompute: 200 surface points + lowest point + centroid → `.npz`.
- [ ] Xử lý object hai phần (top/bottom — affordance vs non-affordance): Phase
  2 chỉ cần phần `top` (bộ train của họ hầu hết một phần); ghi chú object nào
  có `bottom` thật để thêm non-affordance contact penalty nếu cần.
- [ ] Kiểm tra tự động từng object: drop test ổn định trên bàn, không tunnel,
  số contact hợp lý.

## B. Multi-object training

- [ ] `VariantEntityCfg` với ~35 variants; phân bổ env theo object (gốc: 2
  env/object, lặp thêm object khó — scissors, water_body, pitcher, banana,
  mouse, hammer, small_block). Với mjlab nhiều env hơn (512–4096), chia đều +
  oversample object khó.
- [ ] Lưu ý assignment của `VariantEntityCfg` cố định lúc init — chấp nhận
  (giống gốc: object gắn với env), chỉ cần đủ env mỗi object.
- [ ] `nconmax`/`njmax` tính lại theo số env × mesh phức tạp.

## C. Curriculum & sampling

- [ ] Edge-biased sampling vị trí object: 50% uniform, 50% Beta(0.5, 0.5) trên
  cả góc lẫn khoảng cách (port từ train.py gốc).
- [ ] Eval luôn uniform (tách cfg eval khỏi train).

## D. Đánh giá

- [ ] Eval script per-object success rate (grasp 70 bước + lift script),
  bảng xếp hạng object khó như log gốc.
- [ ] So sánh chéo: chạy vài object trùng YCB với số liệu paper (97% sim trên
  bộ train) để định vị chất lượng port.
- [ ] Tune: trọng số finger cho ring/pinky (không có yaw), PD gains, ngưỡng
  impulse — ghi lại mọi chỉnh sửa so với cfg gốc vào doc này.
