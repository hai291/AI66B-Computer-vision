# AI66B-Computer-vision

## Bài toán

Dự án giải quyết cuộc thi Kaggle **[RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification)**.

Mục tiêu: từ ảnh MRI cột sống thắt lưng, phân loại mức độ thoái hóa (degenerative conditions) cho từng đốt sống.

Với mỗi ca chụp (`study_id`), cần dự đoán mức độ nghiêm trọng cho **5 loại bệnh lý (conditions)** tại **5 khe đĩa đệm (levels)** — tổng cộng 25 tổ hợp:

**Conditions**
- Spinal Canal Stenosis
- Left / Right Neural Foraminal Narrowing
- Left / Right Subarticular Stenosis

**Levels**: L1/L2, L2/L3, L3/L4, L4/L5, L5/S1

Mỗi tổ hợp condition–level được phân vào 1 trong 3 lớp mức độ nghiêm trọng:
- `Normal/Mild`
- `Moderate`
- `Severe`

→ Đây là bài toán **multi-label, multi-class classification** (25 nhãn độc lập/study, mỗi nhãn 3 lớp).

### Dữ liệu

| File | Nội dung |
|---|---|
| `train.csv` | Nhãn severity cho từng `study_id` × 25 tổ hợp condition/level |
| `train_series_descriptions.csv` | Map `series_id` → loại chuỗi xung (Sagittal T1, Sagittal T2/STIR, Axial T2) |
| `train_label_coordinates.csv` | Tọa độ (x, y, slice) nơi bác sĩ đánh dấu vị trí đánh giá trên từng ảnh |
| DICOM images | Ảnh MRI gốc, tổ chức theo `study_id/series_id/*.dcm` |

### Metric

Weighted multi-class log loss, trọng số theo severity: `Normal/Mild = 1, Moderate = 2, Severe = 4`, cộng thêm một nhãn phụ `any_severe_spinal` (có tồn tại condition Severe ở Spinal Canal Stenosis hay không) cũng được tính với trọng số cao.

## Model

CNN baseline để phân loại mức độ nghiêm trọng (3 lớp) từ một crop ảnh MRI: xem [model.py](model.py).
