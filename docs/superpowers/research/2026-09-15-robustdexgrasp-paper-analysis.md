# RobustDexGrasp — Phân tích paper (note kỹ thuật, đọc toàn văn + appendix)

> Mục đích: bóc tách **cơ chế zero-shot** của RobustDexGrasp để tái lập trong
> **mjlab / MuJoCo** với **UR5e + rh5dg2 hand**. Tập trung phần *cần copy*
> (pre-grasp + representation + reward + curriculum), không phải dataset.
> Nguồn: đã đọc **nguyên văn arXiv HTML v1 + toàn bộ appendix** (Table VIII–XI).

**Paper:** *Robust Dexterous Grasping of General Objects from Single-view
Perception* — arXiv **2504.05287v1**, 07 Apr 2025.
- **Tác giả:** Hui Zhang¹, Zijian Wu², Linyi Huang², Sammy Christen¹, Jie Song²³
  (¹ETH Zürich, ²HKUST-Guangzhou, ³HKUST Hong Kong). *(paper của nhóm GraspXL/
  FunGrasp/ArtiGrasp — cùng lineage.)*
- Project: https://zdchan.github.io/Robust_DexGrasp/ · Code:
  https://github.com/zdchan/RobustDexGrasp
- Phần cứng: **UR5 + Allegro hand (16 DOF)** + 1 RealSense D435i top-view.
- Simulator: **RaiSim**; PPO; ~30h trên 1× RTX 3090 + 128 CPU (cả teacher+student).
- Kết quả: **97.0%** / 247,786 vật sim (Objaverse), **94.6%** / 512 vật thật —
  **train chỉ 35 vật**.

---

## 0. TL;DR — Vì sao 35 vật → 0-shot

Generalization đến từ **4 thiết kế**, không phải số lượng object:

1. **Hand-centric object representation** — policy nhìn *vector khoảng cách từ
   link tay → điểm gần nhất trên point cloud vật*, tức **quan hệ tiếp xúc cục bộ**,
   **bỏ qua global shape/danh tính vật** → vật lạ có bề mặt cục bộ tương tự là
   xử lý được. Chìa khóa số 1.
2. **Pre-grasp pose initialization** — đặt tay vào tư thế tốt (hướng vào tâm vật,
   ôm cạnh hẹp, cách 25cm) trước khi policy chạy → thu hẹp không gian explore.
   *Không có bước này thì RL rất khó hội tụ.*
3. **Closed-loop RL** target-joint-position, phản ứng theo tiếp xúc thời gian thực.
4. **Teacher–student + mixed curriculum (IL→RL)** — teacher học với point cloud
   đầy đủ + contact GT; student chỉ dùng single-view + proprioception nhiễu, tự
   tái dựng contact bằng LSTM.

> Hệ quả: nếu observation là `joint_pos/joint_vel` thuần thì **không 0-shot**.
> Phải bê nguyên `Φ` hand-centric + pre-grasp init.

---

## 1. Kiến trúc tổng thể (Fig. 2)

```
   TEACHER (RL/PPO, privileged)              STUDENT (IL→RL, deployable)
   obs: a_{t-1}, q (noise-free),   distill   obs: single-view O_0 (tĩnh),
        O_t (point cloud đầy đủ),  ───────►        q̂ nhiễu, c̃ = LSTM(hist)
        c_t = {contact b, impulse f} GT         init = teacher weights
   → target joint positions                   → target joint positions
```
- Contact/impulse reconstruction loss **giữ active suốt** quá trình train student.

---

## 2. Pre-grasp pose initialization (III-B) — ⭐ dễ bỏ sót nhưng quan trọng

- Ngón tay mở **một phần**; `q0` (Allegro, 16 giá trị, phụ lục A4):
  `[0.2,0.6,0.2,0.5, 0.2,0.6,0.2,0.5, 0.2,0.6,0.2,0.5, 1.3,0.0,-0.1,0.2]`.
- **Heading `x`**: từ điểm xuất phát cố định (world frame) → tâm point cloud vật `c`.
- **Palm `y`**: sample **10 hướng** trực giao với `x`, chọn min:
  ```
  l_y = w_length · l_length + ‖q4 − 1.57‖²      (w_length = 5)
  ```
  `l_length` = độ dài hình chiếu point cloud vật theo `y`; `q4` = khớp arm áp chót.
  → ôm vật từ **cạnh hẹp**, tránh singularity.
- Wrist 6D pose từ (`x`,`y`); vị trí = cách `c` **25 cm** theo `x`.
- Giải **IK** ra góc arm; nếu IK fail/self-collision → đặt `y` = hướng gốc arm → `c`,
  init lại.

---

## 3. Hand-centric object representation (III-C1) — ⭐ quan trọng nhất

`Φ(a, q, O) = (d, h, T, Δq)`:

| Ký hiệu | Định nghĩa | Chiều |
|---|---|---|
| `d` | Vector từ **mỗi finger joint → điểm gần nhất** trên point cloud `O` | (#finger joint × 3) |
| `h` | Khoảng cách **thẳng đứng** mỗi joint (arm+hand) → mặt bàn | #joint (arm+hand) |
| `T` | Pose **cổ tay** (vị trí + hướng) | 6 |
| `Δq` | Tracking error = `q_t − a_{t−1}` (góc hiện tại − target trước) | = dim(q) |

Đưa vào policy: `Φ` **+ `q` + `c`**.

> ⚠️ Paper **không cho tổng chiều** cụ thể (chỉ định nghĩa hàm; `L` = #hand link,
> `M` = #arm link để trống số). Muốn con số chính xác phải đọc code.
> `d` là quan hệ **cục bộ, tương đối** → bất biến danh tính vật, ít nhạy thiếu view.

---

## 4. Observation

**Teacher** `s = (a_{t−1}, q, O, c)`:
- `a_{t−1}` action trước · `q` góc khớp arm+hand (không nhiễu)
- `O` point cloud vật **đầy đủ realtime** (privileged)
- `c = {b, f}` **contact nhị phân** + **impulse** mỗi finger link (GT, privileged)

**Student**:
- `Ô_0` point cloud single-view **tĩnh** (chụp trước grasp, không cập nhật sau)
- `q̂` proprioception **nhiễu**
- `c̃ = {b̂, f̂}` **tái dựng bằng LSTM** từ lịch sử (joint state, action).
  *Cơ chế:* action ⇒ suy ra torque actuator; **lệch giữa torque và thay đổi joint
  state ⇒ ngoại lực do tiếp xúc** → không cần cảm biến xúc giác thật.

---

## 5. Action & control

- Output = **target joint positions** cho PD low-level (không torque). Code gốc là
  **residual theo joint state hiện tại**: `pTarget = action·std + q_current`, tính
  một lần mỗi policy step rồi giữ nguyên qua 20 sim step (`Environment.hpp:602-604,
  640`).
- **Allegro 16 + UR5 6 = 22** action. **Policy 5 Hz / PD 100 Hz** → 1 action = **20
  sim step**, sim timestep **0.01 s**.
- **Action delay (III-E2):** train có **ngẫu nhiên trễ** cập nhật action high-level
  để mô phỏng độ trễ inference thật.

> Ánh xạ ta: rh5dg2 **18** + UR5e **6** = **24** action. 5 Hz control ⇒
> `decimation ≈ 20` với `timestep 0.01` (env `view` đang 50 Hz — đổi khi làm task).

---

## 6. Reward (III-C2) — công thức + **trọng số đầy đủ (Table X)**

> ⚠️ **Grasp = closed-loop RL, LIFT = scripted.** RL chỉ tối ưu grasp (vật còn trên
> bàn); nâng vật do quỹ đạo scripted (ngón vẫn closed-loop). Vì thế reward **không
> có term nâng vật**, và penalty vật đứng yên (`w_o=−15`, `w_l=−5`) là **có chủ đích**.

Tổng: `r = r_dis + r_contact + r_height + r_reg`.

**(a) Distance** (theo GraspXL): `r_dis = − Σ_i w_i^d ‖d_i‖²`
**(b) Contact** (thưởng ngón–vật, phạt self/table/arm–object):
```
r_contact = Σ_{i=1..L} b_i (w_i^cd + w_i^fd · f_i^o)      # desired
          − Σ_{j=1..L+M} b_j (w_j^cu + w_j^fu · f_j^u)    # undesired
```
**(c) Height** (phạt link < 2cm mặt bàn): `r_height = Σ w_i^h · log(min{h_i, 0.02})`
**(d) Reg**: `r_reg = w_h‖Ṫ_h‖² + w_o‖Ṫ_o‖² + w_l‖l_o‖ + w_q‖q̇_a‖²`
(`Ṫ_h,Ṫ_o` vận tốc tay/vật; `l_o` dịch chuyển vật; `q̇_a` vận tốc khớp arm)

| Weight | Value | | Weight | Value |
|---|---|---|---|---|
| `w^d` fingertip | **2.0** | | `w^cu` (undesired contact) | **−1.0** |
| `w^d` link khác | **0.5** | | `w^fu` (undesired impulse) | **−0.5** |
| `w^cd` fingertip | **6.0** | | `w_h` (hand vel) | **−1.0** |
| `w^cd` link khác | **1.5** | | `w_o` (object vel) | **−15.0** |
| `w^fd` fingertip | **4.0** | | `w_l` (object displacement) | **−5.0** |
| `w^fd` link khác | **1.0** | | `w_q` (arm joint vel) | **−1.0** |
| `w_re` (contact recon) | **1.0** | | `w_act` (action imit.) | **1.0** |

> Reward **per-link** → khi sang rh5dg2 phải định nghĩa lại tập fingertip / hand
> link / arm link và gán lại `w_i`. Đây là phần adapt tốn công nhất về reward.

---

## 7. Mixed curriculum learning (III-D2)

Hai loss student:
```
L_re  = w_re (‖b̂−b‖² + ‖f̂−f‖²)     # tái dựng contact — GIỮ suốt
L_act = w_act ‖â−a‖²                # bắt chước action teacher — GIẢM dần
```
Chuyển tiếp: `w_act` giảm theo `λ`, RL reward tăng theo `1−λ`, `w_re` cố định:
```
λ = 1.0 − iter_num / 2000          # IL → RL trong ~2000 iteration
```
Student **init từ trọng số teacher**. Đầu ≈ IL thuần (distill nhanh), cuối ≈ RL
thuần (học thích ứng nhiễu/lực).

---

## 8. PPO hyperparameters (Table IX) + mạng

| Param | Value | | Param | Value |
|---|---|---|---|---|
| Epochs | 1.5e4 | | Discount γ | 0.996 |
| Steps / epoch | 70 | | Max grad norm | 0.5 |
| Env steps / episode | 63 | | Value loss coef | 0.5 |
| Batch size | 2000 | | Entropy coef | **0.0** |
| Updates / epoch | 20 | | Hidden layers | 2 |
| Sim timestep | 0.01 s | | Hidden units | 128 |
| Sim steps / action | 20 | | + LSTM encoder | contact recon |

Simulator RaiSim; teacher+student ~30h / RTX3090 + 128 CPU.
**Train objects: 35** — YCB [Calli 2015] + vật scan từ **FunGrasp [Huang 2024]**
(= `rsc/new_training_set` trong repo). *Tiêu chí chọn không nêu tường minh.*

---

## 9. Domain randomization (Table VIII) — chỉ khi train STUDENT

| Variable | Range |
|---|---|
| Friction coefficient | {0.5, 0.6, 0.7, 0.8, 0.9} |
| Hand P gain | [0.9, 1.1] × 600 |
| Hand D gain | [0.9, 1.1] × 20 |
| Arm P gain | [0.5, 1.05] × 1.6e4 |
| Arm D gain | [0.5, 1.05] × 600 |
| Hand joint angle noise | [−0.02, +0.02] rad + GT |
| Arm joint angle noise | [−0.005, +0.005] rad + GT |
| Link position noise | [−0.01, +0.01] m + GT |
| Link orientation noise | [−0.02, +0.02] rad + GT |
| + Action delay | trễ ngẫu nhiên cập nhật action |

> ⚠️ **Mass vật & ngoại lực KHÔNG có trong bảng DR** (chỉ dùng lúc *đánh giá
> robustness*: sim 2.5N random, real đặt vật nặng 250g ⇒ 2.5N). Object pose đặt
> ngẫu nhiên trên bàn (phân phối không nêu). Vật train là **solid**.

---

## 10. Deployment / sim-to-real (III-E, A1) & success metric

- **Point cloud:** lọc điểm thấp hơn mặt bàn; **trung bình 60 frame** (~1.28s) khử
  nhiễu. **Không segmentation model** (không SAM) — chỉ threshold theo bàn.
- **Camera:** RealSense D435i top-view.
- **Success:** nâng vật lên **0.1 m**, giữ ổn định **≥ 3 giây**. Real: mỗi vật **3
  pose ngẫu nhiên**.

---

## 11. Ablation (Table VI–VII) — cái gì thực sự cần

| Setting | Suc. (sim) | Ý nghĩa |
|---|---|---|
| **Ours (student)** | 0.953 | full method |
| Teacher policy | 0.960 | trần trên (privileged) |
| W.o. IL loss | 0.933 | IL giúp nhưng ít |
| W.o. Curriculum (tỉ lệ cố định) | 0.913 | curriculum có đóng góp |
| W.o. RL rewards | 0.907 | RL cần cho robustness |
| **W.o. Priv. learning** (student from scratch) | **0.773** | **teacher là bắt buộc** |
| Real: single-view vs full point cloud | 0.920 vs 0.933 | single-view **gần bằng** → repr hiệu quả |

**Đọc ra:** (1) teacher privileged là thành phần quan trọng nhất (bỏ → tụt 18%).
(2) single-view sparse repr gần bằng full point cloud → không cần tái dựng 3D.

---

## 12. Limitations (paper tự nêu)

- Visual **open-loop** (point cloud tĩnh ban đầu) → kém khi vật **dịch chuyển lớn**.
- **Không nắm được vật đường kính < 1.5 cm** (giới hạn kích thước tay Allegro).
- Đề xuất tương lai: dùng làm low-level skill + high-level (VLM/ngôn ngữ).

---

## 13. Ánh xạ sang mjlab + UR5e/rh5dg2 — việc phải làm

| Thành phần | Transfer? | Việc ở ta |
|---|---|---|
| Target-joint-position control | ✅ | Có `JointPositionActionCfg`; đổi decimation→~20, timestep→0.01 |
| Pre-grasp init (IK + 25cm + palm sampling) | ⚠️ vừa | Cần IK cho UR5e + logic heading/palm; rh5dg2 `q0` mở ngón riêng |
| `Φ`: `d` (link→closest PC point) | ⚠️ viết mới | Xác định tập finger link rh5dg2; cần point cloud vật |
| `Φ`: `h`, `T`, `Δq` | ✅ dễ | Có từ entity data + `TABLE_TOP_Z` đã biết |
| Contact `c={b,f}` (teacher GT) | ⚠️ | MuJoCo cho contact/impulse per-geom → map per-link |
| Reward 4 term + Table X weights | ⚠️ map lại | Định nghĩa fingertip/link rh5dg2, gán `w_i` |
| Teacher–student + IL loss + curriculum | ⚠️ lớn | rsl-rl (mjlab) **chưa có** → mở rộng runner hoặc train 2 pha tay |
| Single-view point cloud + LSTM contact | ⚠️ lớn | mjlab thiên state-obs; cần render depth hoặc proxy |
| Domain randomization (Table VIII) | ✅ | event/randomization mjlab (friction, PD gain, noise, delay) |
| 35 train objects | ✅ | Lightwheel-YCB (physics chuẩn) train; GSO/Objaverse lọc làm eval unseen |

**Thứ tự triển khai đề xuất (MVP → đầy đủ):**
1. **Teacher state-based MVP:** pre-grasp init + obs `Φ=(d,h,T,Δq)` + contact GT
   MuJoCo + reward 4 term (weights Table X) + PPO. 1 vật cố định → xác nhận grasp
   (metric: nâng 0.1m giữ 3s). *Bỏ single-view/LSTM giai đoạn này.*
2. **Object randomization** (pose + nhiều vật Lightwheel-YCB) → đo generalize.
3. **Student**: single-view point cloud + LSTM contact reconstruction.
4. **Curriculum IL→RL** (`λ = 1 − iter/2000`) + domain randomization Table VIII.

---

## 14. Điểm cần đọc CODE để chốt (chưa có trong paper)

- Tổng chiều `Φ` và observation vector; tập link `L` (hand), `M` (arm) chính xác.
- Map contact/impulse per-geom (MuJoCo) → per-link `b, f`.
- Danh sách 35 vật cụ thể + scale (trong `rsc/new_training_set`).
- Range randomization **mass vật** (không có trong Table VIII).
- Kiến trúc LSTM encoder (kích thước history, hidden).
- License asset: RobustDexGrasp/Lightwheel-YCB **CC BY-NC** (phi thương mại).

---

## 15. Tham chiếu vật thật (Table XI) — hữu ích cho DR & chọn object

512 vật thật, 12 nhóm; tổng: **mass 7–610 g**, **scale 35×30×15 → 400×200×130 mm**.
Nhóm (success): Toy Cars 0.979 · Snacks 0.974 · Other Daily 0.971 · Bottles&Boxes
0.970 · Building Blocks 0.963 · Fruit&Veg 0.962 · Deformable 0.957 · Wooden 0.940 ·
Animal 0.907 · Picnic 0.902 · Real Tools 0.893 · Tool Models 0.875.
→ **Tool/thin/heavy khó nhất**; dùng dải mass/scale này để set randomization vật.

---

## 16. Nguồn

- HTML (đã đọc toàn văn): https://arxiv.org/html/2504.05287v1
- PDF: https://arxiv.org/pdf/2504.05287 · Project: https://zdchan.github.io/Robust_DexGrasp/
- Code+dataset: https://github.com/zdchan/RobustDexGrasp
