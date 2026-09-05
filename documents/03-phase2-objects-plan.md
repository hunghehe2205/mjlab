# Object curriculum — single → 5 → 35

Tên file giữ lại từ roadmap cũ. Trong code, task 5 object mang hậu tố
`Phase1`; tài liệu này mô tả toàn bộ bước scale object sau khi single-object
ổn định.

## Điều kiện vào

Không train dài 5 object chỉ vì task đã build được. Điều kiện vào là cổng R1
ở [09 §6](09-r1-plan.md) cộng smoke from-scratch ở 09 §7:

- 5 checkpoint liên tiếp có mean scripted lift ba seed ≥ 60%, mean của 5
  checkpoint ≥ 70%, không cú tụt liền kề quá 15 điểm;
- `thumb_yaw_last` trung bình cửa sổ ≥ 0.6;
- `object_displacement_last` không cao hơn đầu cửa sổ quá 2 cm, `frac_drop`
  không tăng; `frac_tipped` chỉ báo cáo;
- smoke from-scratch 300 iter của cấu hình thắng có contact trước iter 200,
  hoặc pipeline được ghi rõ là phụ thuộc checkpoint baseline.

Baseline `8yelo0qc` chưa qua điều kiện này: `model_1400` đạt 60.4% ba seed
với 38% vật lật, đỉnh đơn 80.5% ở iter 1450 nhưng tụt còn 14.1% tại iter 1500.
Hai smoke run normalization không bootstrap được contact và không được dùng để
warm-start.

## Pipeline đã triển khai

- [x] Nhập 35 object của `new_training_set`.
- [x] Dùng `top_watertight_tiny.obj` và tạo `.npz` gồm 200 surface points,
  normals, centroid, lowest point.
- [x] Tạo mesh object MJCF với freejoint, material và convex-hull collision.
- [x] `VariantEntityCfg` gán object cố định theo world.
- [x] Giữ phân bổ weighted 88 slot cho full task.
- [x] Sampling edge-biased Beta(0.5, 0.5) cho train.
- [x] Eval có thể chọn object và báo lift success theo object.
- [x] Cờ `--seed` để so checkpoint trên cùng tập pose (working tree).
- [x] Cấu hình curriculum 5 object trong code:
  potted meat can, tomato soup can, tuna fish can, sugar box, pudding box.
- [ ] Đổi `PHASE1_OBJECT_NAMES` sang bộ vật đứng khi bắt đầu Mốc 1:
  potted_meat_can, tomato_soup_can, master_chef_can, sugar_box,
  mustard_bottle. Lý do: tuna_fish_can cao 3.3 cm và pudding_box cao 3.8 cm,
  rộng 13.5 cm, là vật phẳng cần pinch từ trên và bị các term table chi phối;
  để batch 2 cùng gelatin_box, banana, small_tape.

| Object | dài × rộng × cao (cm) | Vai trò |
| --- | --- | --- |
| potted_meat_can | 10.1 × 5.9 × 8.3 | baseline đã học |
| tomato_soup_can | 6.6 × 6.7 × 10.1 | trụ nhỏ |
| master_chef_can | 10.2 × 10.2 × 14.0 | trụ to, thử độ mở bàn tay |
| sugar_box | 4.8 × 9.3 × 17.6 | hộp cao, cạnh mỏng |
| mustard_bottle | 9.6 × 6.6 × 18.9 | dáng bất đối xứng |

## Phần còn thiếu

- [ ] Drop-test tự động toàn bộ 35 object: resting pose, penetration, contact
  count và tunneling.
- [ ] Re-evaluate `nconmax`/`njmax` dưới tải multi-object thực tế.
- [ ] Phân loại object có mesh bottom/non-affordance và quyết định có cần
  penalty riêng hay không.
- [ ] Collision decomposition tốt hơn một convex hull nếu object lõm gây sai
  hình học nắm.
- [ ] Train và eval curriculum 5 object.
- [ ] Train và eval full 35 object.

## Kế hoạch thực nghiệm

### Mốc 1 — five-object smoke

Chỉ bắt đầu sau khi teacher single-object qua cổng. Warm-start checkpoint tốt
nhất, giữ PPO/reward/action/obs cố định và chỉ đổi object set. 1024 env chia
đều, khoảng 205 env mỗi vật. Chạy smoke 100–300 iter để kiểm:

- mỗi object thực sự xuất hiện với số env mong đợi;
- reset/pre-grasp không có object-specific failure;
- contact và reward không bị một object dễ chi phối;
- không tràn contact buffer hoặc phát sinh NaN.

### Mốc 2 — five-object train

Train 1000–1500 iter. Đánh giá checkpoint mỗi 50 iteration, ít nhất ba seed.
Báo cả macro-average theo object và bảng per-object; không chỉ báo average
theo env vì sampling có trọng số. Cổng qua: mean ≥ 70%, worst-object ≥ 50%,
5 checkpoint liên tiếp ổn định, displacement không tăng. Vật tụt thì
oversample bằng `assignment`, không thêm reward riêng.

### Mốc 3 — full 35 object

Warm-start từ checkpoint 5 object tốt nhất. Giữ assignment cố định theo env,
oversample object khó theo weight hiện có. Chỉ thay collision representation
hoặc reward cho một nhóm object sau khi có failure taxonomy từ eval.

## Tiêu chí đánh giá

- lift success cuối pha scripted lift;
- mean và worst-object success;
- spread theo seed/checkpoint;
- thumb yaw, contact-body count, horizontal load và push từ metric online
  (09 §3) và probe offline;
- `frac_tipped`, `frac_drop` từ eval;
- object displacement/out-of-workspace rate;
- throughput và đỉnh số contact.

Số 97% simulation hoặc 94.6% real-world của paper không phải cổng trực tiếp
cho phase này: hệ hiện tại khác robot, simulator, collision mesh và mới có
teacher. Chúng chỉ là mốc tham khảo sau khi hoàn tất cùng phạm vi pipeline.
