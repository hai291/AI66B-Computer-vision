# AI66B Computer Vision — Phân loại thoái hóa cột sống thắt lưng trên MRI

Dự án học tập sử dụng dữ liệu từ cuộc thi [RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification). Mục tiêu là phân tích ảnh MRI cột sống thắt lưng và dự đoán mức độ hẹp/thoái hóa tại từng **khe đĩa đệm**. Repository tập trung vào tìm hiểu dữ liệu, xử lý ảnh DICOM và xây dựng mô hình phân loại.

> Dự án phục vụ học tập, nghiên cứu; kết quả mô hình không thay thế đánh giá của bác sĩ.

## 1. Bài toán

**Đầu vào:** Một ca chụp MRI (`study_id`) gồm một hoặc nhiều chuỗi ảnh (`series_id`), mỗi chuỗi chứa nhiều lát cắt DICOM.

**Đầu ra:** Với mỗi ca chụp, dự đoán mức độ nghiêm trọng cho **5 vị trí đánh giá × 5 khe đĩa đệm = 25 tổ hợp**:

| Vị trí đánh giá (`condition`) | Ý nghĩa |
| --- | --- |
| `spinal_canal_stenosis` | Hẹp ống sống |
| `left_neural_foraminal_narrowing` | Hẹp lỗ liên hợp bên trái |
| `right_neural_foraminal_narrowing` | Hẹp lỗ liên hợp bên phải |
| `left_subarticular_stenosis` | Hẹp ngách bên trái |
| `right_subarticular_stenosis` | Hẹp ngách bên phải |

Năm khe đĩa đệm (`level`): **L1/L2, L2/L3, L3/L4, L4/L5, L5/S1**.

Mỗi tổ hợp `condition`–`level` thuộc một trong **3 lớp**: `Normal/Mild`, `Moderate`, `Severe`. Mô hình có thể xuất ra 3 xác suất cho từng tổ hợp, tức tối đa **25 × 3 = 75 giá trị xác suất** cho một ca chụp; trong mỗi bộ ba, tổng xác suất bằng 1. Đây là **bài toán phân loại đa đầu ra**, với mỗi đầu ra là bài toán phân loại 3 lớp. Con số 75 là số xác suất, **không phải 75 lớp severity khác nhau**.

Ví dụ: `spinal_canal_stenosis_l3_l4` là mức độ hẹp ống sống tại khe L3/L4, không phải nhãn của một ảnh DICOM đơn lẻ.

## 2. Nguồn và cấu trúc dữ liệu

Dữ liệu gồm **ảnh MRI định dạng DICOM (`.dcm`)** và **các bảng chú giải định dạng CSV**. Ảnh được tổ chức theo ca chụp → chuỗi ảnh → lát cắt:

```text
data/
├── train.csv
├── train_series_descriptions.csv
├── train_label_coordinates.csv
├── sample_submission.csv
├── train_images/
│   └── <study_id>/
│       └── <series_id>/
│           └── <instance_number>.dcm
├── test_series_descriptions.csv  # nếu được cung cấp trong môi trường test
└── test_images/                 # dữ liệu test có thể được cấp khi chấm notebook
```

> Thư mục `data/` ở đây là cách tổ chức **trên máy của nhóm**. Trên Kaggle Notebook, dữ liệu nằm tại `/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/`. Bộ test của cuộc thi dùng cơ chế test ẩn; không nên giả định máy cá nhân có đủ file test hoặc nhãn test.

### 2.1. `train.csv` — nhãn mức độ nghiêm trọng

| Trường | Ý nghĩa |
| --- | --- |
| `study_id` | Mã ca chụp; dùng để liên kết nhãn với các chuỗi ảnh. |
| `<condition>_<level>` | Một trong 25 cột nhãn, ví dụ `spinal_canal_stenosis_l1_l2`; giá trị là `Normal/Mild`, `Moderate` hoặc `Severe`. |

Mỗi hàng đại diện cho **một ca chụp**, không phải một lát ảnh. Một số nhãn có thể thiếu; cần kiểm tra giá trị thiếu và chỉ tính loss trên các nhãn có thật. Không tự suy diễn nhãn thiếu thành `Normal/Mild`.

### 2.2. `train_series_descriptions.csv` — thông tin chuỗi ảnh

| Trường | Ý nghĩa |
| --- | --- |
| `study_id` | Mã ca chụp. |
| `series_id` | Mã chuỗi ảnh thuộc ca chụp. |
| `series_description` | Loại chuỗi MRI, chẳng hạn `Sagittal T1`, `Sagittal T2/STIR`, `Axial T2`. |

Một `study_id` có thể có **nhiều** `series_id`. `series_description` hỗ trợ chọn loại ảnh phù hợp cho từng vị trí cần đánh giá; không phải nhãn severity.

### 2.3. `train_label_coordinates.csv` — vị trí vùng được đánh dấu

| Trường | Ý nghĩa |
| --- | --- |
| `study_id` | Ca chụp chứa vị trí được đánh dấu. |
| `series_id` | Chuỗi ảnh chứa lát cắt liên quan. |
| `instance_number` | Số lát cắt; cũng là tên file `<instance_number>.dcm` trong chuỗi tương ứng. |
| `condition` | Loại bệnh lý/vị trí đánh giá. |
| `level` | Khe đĩa đệm, ví dụ `L3/L4`. |
| `x`, `y` | Tọa độ tâm vùng được đánh dấu trên lát cắt. |

Bảng này giúp **định vị lát cắt và crop vùng quan tâm**. Severity dùng để huấn luyện nằm trong `train.csv`; không nên coi bảng tọa độ là danh sách đầy đủ mọi lát cắt hoặc mọi nhãn severity.

### 2.4. Ảnh DICOM (`train_images/`)

Mỗi file `<instance_number>.dcm` là một lát cắt MRI. Có thể nối thông tin giữa các bảng theo chuỗi:

```text
train.csv: study_id
    ↓
train_series_descriptions.csv: study_id → series_id, series_description
    ↓
train_label_coordinates.csv: study_id + series_id → instance_number, condition, level, x, y
    ↓
train_images/<study_id>/<series_id>/<instance_number>.dcm
```

Không nên xem mọi lát cắt trong cùng `study_id` là cùng một mẫu độc lập khi chia train/validation; các lát cùng ca chụp phải ở cùng một tập để tránh rò rỉ dữ liệu.

### 2.5. `sample_submission.csv` và dữ liệu test

`sample_submission.csv` mô tả định dạng dự đoán của cuộc thi:

| Trường | Ý nghĩa |
| --- | --- |
| `row_id` | Kết hợp `study_id`, `condition` và `level`, ví dụ `12345_spinal_canal_stenosis_l3_l4`. |
| `normal_mild`, `moderate`, `severe` | Xác suất dự đoán cho ba mức độ; mỗi hàng cần có tổng xác suất bằng 1. |

File mẫu vẫn hữu ích để hiểu **định dạng đầu ra**, dù nhóm chỉ thực hiện dự án học tập và không nộp lên cuộc thi. Dữ liệu test không có cột nhãn severity công khai như `train.csv`.

## 3. Đánh giá mô hình

Cuộc thi sử dụng **sample-weighted multiclass log loss**; nhãn `Normal/Mild`, `Moderate`, `Severe` lần lượt có trọng số **1, 2, 4**. Điểm tổng hợp các nhóm bệnh lý và có thêm thành phần `any_severe_spinal` liên quan đến việc có hẹp ống sống mức `Severe` ở ít nhất một trong năm khe. Vì vậy, khi so sánh với điểm Kaggle cần dùng đúng [định nghĩa metric của cuộc thi](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/overview/evaluation), thay vì chỉ lấy accuracy hoặc log loss thông thường.

Khi huấn luyện/đánh giá trên dữ liệu train:

- Chia tập theo `study_id` để cùng một ca chụp không xuất hiện ở cả train và validation.
- Bỏ qua các mục có nhãn bị thiếu khi tính loss; theo dõi số lượng mẫu ở từng lớp vì mức `Severe` thường ít hơn.
- Đánh giá riêng theo nhóm bệnh lý và báo cáo metric dùng trên tập validation; không coi điểm của một mô hình crop 3 lớp là điểm toàn bộ cuộc thi.

## 4. Định hướng baseline

Một baseline khả thi là: đọc lát DICOM được chỉ định → dùng `(x, y)` để crop vùng quan tâm → chuẩn hóa ảnh → huấn luyện CNN dự đoán 3 mức severity cho **một tổ hợp condition–level**. Để đáp ứng đầy đủ bài toán của cuộc thi, cần dự đoán và tổng hợp cả 25 tổ hợp cho mỗi `study_id`, đồng thời xử lý các trường hợp thiếu tọa độ/nhãn.

Nếu repository có `model.py`, xem file đó để biết kiến trúc và phạm vi mà nhóm **đã thực sự triển khai**. Phần mô tả baseline ở trên là định hướng xử lý dữ liệu, không phải tuyên bố về kết quả hay tính năng đã hoàn tất.

## 5. Đọc bảng dữ liệu trong VS Code / Jupyter

Đăng nhập Kaggle, chấp nhận điều khoản truy cập dữ liệu, sau đó tải các CSV cần dùng từ [trang Data](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data) vào thư mục `data/`. Có thể dùng Kaggle CLI để tải **riêng** `train.csv`, tránh tải cả bộ ảnh khi mới khám phá nhãn:

```bash
kaggle competitions download rsna-2024-lumbar-spine-degenerative-classification -f train.csv -p data
```

Nếu file nhận được là `.zip`, hãy giải nén để có `data/train.csv`. Trong notebook chạy bằng VS Code trên máy cá nhân:

```python
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")  # đổi thành đường dẫn tuyệt đối nếu notebook có working directory khác
train_path = DATA_DIR / "train.csv"

print("Đường dẫn:", train_path.resolve())
print("Tồn tại:", train_path.exists())

train = pd.read_csv(train_path, dtype={"study_id": "string"})
print("Kích thước:", train.shape)
display(train.head())
```

Nếu `Tồn tại: False`, kiểm tra `Path.cwd()` và sửa `DATA_DIR` thành đúng nơi đã lưu CSV. Đường dẫn `/kaggle/input/...` chỉ áp dụng khi notebook chạy **trong môi trường Kaggle**.

Để khảo sát các nhãn thiếu:

```python
label_columns = train.columns.drop("study_id")
print("Số cột nhãn:", len(label_columns))  # dự kiến 25
print(train[label_columns].isna().sum().sort_values(ascending=False).head(10))
print(train[label_columns].stack().value_counts())
```

## 6. Sử dụng dữ liệu và trích dẫn

Dataset có [quy định riêng của cuộc thi](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/rules): sử dụng cho mục đích phi thương mại/nghiên cứu, giáo dục theo điều khoản áp dụng; **không đăng ảnh DICOM hoặc bản sao dữ liệu cuộc thi lên GitHub**, không phân phối lại dữ liệu và không cố gắng tái nhận dạng người bệnh. Người dùng tải dữ liệu trực tiếp từ Kaggle và chấp nhận điều khoản ở nguồn.

Trích dẫn dataset theo yêu cầu của ban tổ chức: **“RSNA 2024 Lumbar Spine Degenerative Classification Challenge.”**

Nguồn tham khảo:

- [Kaggle — mô tả và tải dữ liệu](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data)
- [Kaggle — cách tính điểm](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/overview/evaluation)
- [Kaggle — điều khoản sử dụng dữ liệu](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/rules)