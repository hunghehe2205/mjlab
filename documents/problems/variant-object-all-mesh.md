# VariantEntityCfg chỉ biến thiên mesh geom — primitive phải đồng nhất

**Ngữ cảnh:** Plan §B muốn object entity dùng `VariantEntityCfg` để mỗi world
một object khác nhau ("mỗi world một mesh"). §A tạo 5 object: box + cylinder
(primitive geom) + 3 YCB mesh.

**Phát hiện (test thực tế):** `VariantEntityCfg` yêu cầu **primitive geom giống
hệt nhau** giữa các variant; **chỉ geom mesh mới được phép khác nhau** per-world.
- 3 mesh YCB làm variant chung → OK (`nmesh=3`, variant_metadata dựng được).
- box + cylinder (type box vs cylinder) → FAIL: "primitive geom differs".
- trộn box + mesh → FAIL: "primitive geoms must match; only mesh geom counts".

**Hệ quả:** Muốn train multi-object trong 1 scene (1 object entity, mỗi world 1
object) thì **mọi object phải là mesh geom**.

**Lựa chọn:**
- **A (đề xuất):** Mesh-hóa cả box + cylinder (xuất `.obj` từ trimesh primitive)
  → cả 5 object đều mesh → `VariantEntityCfg` gồm cả 5. Đồng nhất pipeline;
  box convex-hull = box chính xác, cylinder ≈ cylinder (faceted, chấp nhận).
- **B:** Chỉ dùng 3 YCB mesh cho variant; box/cylinder giữ procedural, chỉ dùng
  làm single-object debug (không vào variant). Phase 1 train 3 object.

**Đã chốt & làm: Option A.** box + cylinder xuất `.obj` (trimesh) →
`assets/{box,cylinder}/collision.obj`; cả 5 object đồng nhất mesh geom.
`get_phase1_variant_cfg()` dựng `VariantEntityCfg` cả 5 (test: `nmesh=5`,
variant_metadata OK).

**Còn lại (→ §C):** wire variant vào scene cần đặt object đúng mặt bàn theo
`lowest_point` **từng world** (mỗi variant khác nhau) — thuộc reset/pose-
sampling §C. §B skeleton vẫn single-object (`potted_meat_can`) để chạy được.
