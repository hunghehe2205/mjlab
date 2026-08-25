# DexGrasp — object rời bàn gây physics divergence

## Trạng thái

**Đã containment case tái lập (2026-08-25).** Termination
`object_out_of_workspace` kết thúc episode khi object rời task workspace, trước
khi quỹ đạo off-table đạt vận tốc quay không ổn định. Smoke train cùng cấu hình
tái lập chạy đủ 20/20 iteration, `Episode_Termination/nan = 0` toàn bộ và không
sinh NaN dump mới.

Instability high-spin của MJWarp vẫn còn; termination ngăn đường dẫn đã quan sát
trong train chứ không sửa backend physics.

## Cách tái lập (lỗi gốc)

Chạy CPU với cohort mặc định, 8 worlds, seed mặc định `42`, policy chưa train:

```sh
uv run train Mjlab-DexGrasp-UR5eRH5DG2 \
  --gpu-ids None \
  --env.scene.num-envs 8 \
  --agent.num-steps-per-env 8 \
  --agent.max-iterations 20 \
  --agent.save-interval 10 \
  --agent.experiment-name dexgrasp_teacher_smoke \
  --agent.run-name smoke_20it \
  --agent.logger tensorboard \
  --agent.upload-model False \
  --enable-nan-guard True \
  --log-root logs/smoke
```

Kết quả ngày 2026-08-25 (trước khi sửa):

- Run `2026-08-25_09-10-20_smoke_20it`: NaN/Inf ở physics step `1888`, world
  `4`.
- Chạy lại cùng cấu hình nhưng tắt MultiCCD ở runtime
  (`--env.sim.mujoco.disableflags "('multiccd',)"`): NaN/Inf ở step `1886`,
  vẫn world `4`.
- Phân bổ variant cố định cho 8 worlds với seed/cấu hình này là
  `master_chef_can`, `banana`, `pitcher_base`, `scissors`, **`hammer`**,
  `mouse`, `off_water_body`, `small_block`; vì vậy world `4` là `hammer`.

## Quan sát từ NaN dump

NaN guard lưu 100 physics states ngay trước lỗi trong
`/tmp/mjlab/nan_dumps/nan_dump_*.npz`.

Mặt bàn có phạm vi local `x ∈ [-0.60, 0.60]`, `y ∈ [-1.30, -0.20]`. Phân tích
lại qpos/qvel từng state (layout `QPOS | QVEL | MOCAP_POS | MOCAP_QUAT`,
object free joint ở `qpos[24:31]`):

- Object **không trượt khỏi bàn mà bị hất bay**: tại step 1788 nó đã ở
  `(-0.97, -2.17, z=1.42)` — cao hơn mặt bàn ~0.65 m — kèm `|v| ≈ 2 m/s`,
  `|ω| ≈ 36 rad/s`. Quỹ đạo là parabol: lên tới `z = 1.46` quanh step 1795,
  rơi xuống, chạm sàn (`z = 0`) quanh step 1850.
- Sau cú chạm sàn, object bật lên với `|ω| ≈ 101 rad/s`. Khi đã bay tự do và
  không còn contact, `|ω|` vẫn tăng 111 → 130 → 212 → 675 → 1554 → 9594
  rad/s, rồi state bùng nổ và thành NaN/Inf ở step 1888.

## Nguyên nhân gốc (xác nhận bằng replay)

Replay bằng đúng per-world model tái hiện NaN từ state step 1870 sau 18 physics
steps ở timestep 10 ms. Sweep free-body không contact, khởi tạo angular
velocity 100 rad/s trong 0.5 s cho kết quả:

- **Native MuJoCo:** 35/35 object finite ở timestep 10, 5 và 2 ms.
- **MJWarp:** 13/35 finite ở 10 ms, 18/35 ở 5 ms và 33/35 ở 2 ms.
- `hammer` đơn lẻ, không dùng `VariantEntityCfg`, vẫn diverge; đây không phải
  lỗi per-world variant assignment.

=> Đây là instability trong free-body angular dynamics của MJWarp khi object
quay nhanh. Ground impact là trigger tạo `|ω|` lớn, nhưng energy tiếp tục tăng
trong free flight; mesh-plane contact không phải điều kiện cần. Giảm timestep
chỉ trì hoãn, không loại bỏ lỗi. Asset, contact parameters và MultiCCD không
phải nguyên nhân gốc.

## Kiểm tra physics object

- 35/35 object compile và settle finite trên surface trong 5 s ở timestep 10
  ms.
- Mass/inertia per-world khớp compile độc lập; sai số inertia lớn nhất quan sát
  < `4.3e-11 kg·m²`.
- `hammer` có mass 0.24875 kg và principal-inertia ratio ~13.1, lớn nhất cohort.
  Đây là hình học dài/mảnh hợp lệ nhưng nhạy với high-spin instability.

=> Mesh, mass, inertia và variant pipeline đúng ở điều kiện bình thường. Policy
ngẫu nhiên có thể hất object nhẹ ra khỏi bàn; sau đó floor impact đưa backend
vào miền angular velocity không ổn định.

## Giới hạn của NaN dump

Raw `qpos/qvel` trong `.npz` là state thật của world lỗi. Tuy nhiên `.mjb` hiện
được lưu từ host template model, không sync các field per-world của variant.
Trong dump world 4, `.mjb` mang inertia template của `master_chef_can`, không
phải inertia của `hammer`; replay/visualization trực tiếp bằng file này có thể
sai mesh và inertia.

## Sửa

- Termination mới `object_out_of_workspace`
  (`src/mjlab/tasks/dexgrasp/mdp/terminations.py`): terminate khi object root
  ra khỏi khối box quanh bàn (frame env-local).
- Bounds (`OBJECT_WORKSPACE_BOUNDS` trong
  `config/ur5e_rh5dg2/env_cfgs.py`): xy = mép bàn + 0.10 m,
  `z ∈ [mặt bàn − 0.05, mặt bàn + 0.50]`. Vùng spawn object nằm sâu trong
  box, nên không có false positive cho grasp/lift hợp lệ; object rời box là
  episode không cứu được.
- Đăng ký trong `dexgrasp_ur5e_rh5dg2_env_cfg` cùng `hand_below_table`/`nan`.
- `ObjectDisplacement.reset()` đọc trực tiếp pose vừa ghi trong `qpos`/mocap,
  không đọc derived pose cũ trước `sim.forward()`; displacement reward nay lấy
  đúng pose đầu episode làm baseline.
- W&B ghi thêm peak object linear/angular speed theo episode. Notebook Kaggle
  lưu NaNGuard dump, resolved config, git state và metadata lệnh chạy vào
  diagnostics artifact của cùng run.

## Verify

- Smoke train sau cả termination và reward reset fix: đủ 20/20 iteration,
  `Episode_Termination/nan = 0` mọi iteration, không dump mới.
  `Episode_Termination/object_out_of_workspace` xuất hiện thường xuyên như kỳ
  vọng với policy chưa train; episode reset sạch thay vì diverge.
- Toàn bộ suite `uv run pytest tests/ -k dexgrasp`: 232 passed.
- Termination tests gồm regression từ state `hammer` trong dump: 4 passed.
- `make check`: pass.

Notebook chạy lại trên Kaggle nằm tại
`notebooks/dexgrasp_kaggle_train.ipynb`; notebook có smoke run bắt buộc trước
full train và cell evaluation lift success theo từng object.

## Nhận định còn lại

- Smoke 20 iterations xác nhận containment của case seed 42, chưa chứng minh
  train dài tuyệt đối an toàn. High spin xảy ra ngay trong workspace vẫn có thể
  diverge trước khi termination được đánh giá ở policy boundary.
- Nếu sau này nâng cấp mujoco_warp, cần chạy lại free-spin regression; issue có
  thể báo upstream bằng minimal repro không contact.
- NaN guard nên lưu model đã sync theo từng dumped world hoặc ghi rõ template
  model để tránh phân tích nhầm variant.
- Các object variant khác (Phase 2) nên qua drop test tự động tương tự
  (đã có trong checklist `documents/03-phase2-objects-plan.md`).
