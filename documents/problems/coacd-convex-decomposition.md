# coacd không có sẵn — convex decomposition cho object collision

**Ngữ cảnh:** Plan Phase 1 §A và Phase 2 §A yêu cầu mesh → MJCF bằng convex
decomposition với `coacd`. Kiểm tra: `coacd` chưa cài trong env (`trimesh 4.8.3`
thì có). Cài `coacd` là thêm dependency mới (sửa `pyproject.toml`).

**Vì sao chưa chặn Phase 1:** Object gốc dùng `top_watertight_tiny.obj` làm
collision trực tiếp (RaiSim hỗ trợ mesh lõm). MuJoCo mesh-collision chỉ lấy
**convex hull**. Với object gần-lồi của Phase 1 (box, cylinder, hộp/lon YCB),
convex hull ≈ hình thật → dùng mesh trực tiếp là đủ, chưa cần coacd.

**Quyết định:**
- Phase 1: collision = convex hull của mesh (không coacd). Chỉ chọn object
  gần-lồi (potted_meat_can, tomato_soup_can, tuna_fish_can, sugar_box, +
  box/cylinder procedural). Banana để cân nhắc — hơi lõm nhẹ.
- Phase 2: object lõm thật (mug, scissors, hammer, pitcher) cần coacd → khi đó
  mới thêm dependency và pipeline decomposition.

**Cần người dùng quyết khi tới Phase 2:** thêm `coacd` vào `pyproject.toml`
(hoặc dùng VHACD của trimesh). Chưa làm gì với deps ở Phase 1.
