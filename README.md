# VGEC Factory

Hệ thống Python sinh dữ liệu sửa lỗi ngữ pháp tiếng Việt theo quy ước nhãn của bạn.
Profile local mặc định xuất **300 mẫu nhãn 2**, ID **120001–120300**; runner Modal
mặc định **10 mẫu** và cho đổi bằng `--count`. Hệ thống dùng:

- Generator: `Qwen/Qwen3.5-4B`.
- Generator bổ sung tùy chọn: `Qwen/Qwen2.5-7B-Instruct`.
- Verifier độc lập: `Qwen/Qwen3-4B-Instruct-2507`.
- Reverse Error Injection, Label-Conditioned Few-Shot, kiểm nhãn mù,
  kiểm phép sửa, khử trùng và Hard Example Mining từ lỗi classifier.

## Cấu hình trên máy hiện tại

Đã chọn Transformers local. Profile `configs/local-4bit.json` tham chiếu
`D:/Project/VLSP_VGEC_2026/data/VGEC 2026_v2.json` làm tập bảo vệ chống rò rỉ.
Đây là file gốc **457 câu**, không phải file split 90 câu ở
`data/preprocced_data/dev.json`. Chưa tự chọn lại split hoặc dùng câu nào trong
file 457 câu làm seed/few-shot.

```powershell
.\scripts\run-local.ps1 -Command doctor
.\scripts\run-local.ps1 -Command run
```

Runner dùng `.venv` trong Gen, cache `.hf_cache` trong Gen và tắt ghi bytecode.
Môi trường này tái sử dụng package CUDA/PyTorch của venv dự án cũ qua file `.pth`
chỉ đọc; các package nâng cấp cài riêng trong Gen. Không di chuyển/xóa venv cũ
nếu vẫn dùng cấu hình chia sẻ này. Trên máy khác, cài dependencies local đầy đủ.
Revision của hai model đã được ghim trong config.

Trạng thái cài dependency trên máy này nằm ở `outputs/local-setup/status.json`,
log ở `outputs/local-setup/install.log` và `install.err.log`. Chỉ khi trạng thái
`runtime_ready` mới chạy sinh dữ liệu. Nếu tải mạng bị ngắt, chạy lại:

```powershell
.\.venv\Scripts\python.exe -B scripts/setup-local.py
```

Script tiếp tục wheel đã tải một phần, kiểm SHA-256 của PyPI, cài trong Gen rồi
kiểm tra một phép tính CUDA NF4 nhỏ. Kiểm tra này **không phải** inference Qwen.
Trọng số hai model khoảng 17,4 GB chưa tải xong; lần chạy `run` đầu tiên sẽ tải
qua Hugging Face vào cache riêng. Tốc độ mạng thực đo thấp/dao động trong phiên
cấu hình, nên chưa chạy pipeline 300 câu hoặc báo chất lượng model.

Sau khi trọng số sẵn sàng, có thể chạy `scripts/smoke-local.py` bằng Python của
Gen với `HF_HOME` trỏ `.hf_cache` để kiểm thử sinh/kiểm một cặp câu trên model thật.

Đọc/kiểm tra file gốc trực tiếp:

```powershell
python -m vgec inspect-data "D:\Project\VLSP_VGEC_2026\data\VGEC 2026_v2.json"
```

Đầu vào gold hỗ trợ mảng JSON hoặc JSONL, `Fixed Sentence: []` cho nhãn 0
(nghĩa là giữ nguyên Input) và nhiều câu sửa hợp lệ cho nhãn lỗi. Không sửa file
gốc, không bỏ đáp án thay thế. Khử trùng kiểm tra **mọi** reference. Hợp đồng
đầu ra synthetic vẫn nghiêm ngặt: đúng một câu sửa, nhãn 2 và ID liên tiếp.

## Chạy trên Modal

Từ PowerShell tại `D:\Project\Gen`, cài Modal CLI và đăng nhập một lần:

```powershell
.\.venv\Scripts\python.exe -m pip install "modal>=1,<2"
.\.venv\Scripts\python.exe -m modal setup
```

Sau đó chạy pipeline trên GPU A100 40 GB. Modal runner mặc định sinh **10 câu**:

```powershell
.\.venv\Scripts\python.exe -m modal run scripts\modal_vgec.py --corpus "D:\Project\VLSP_VGEC_2026\data\VGEC 2026_v2.json"
```

Đổi số câu bằng `--count`, ví dụ sinh 30 câu:

```powershell
.\.venv\Scripts\python.exe -m modal run scripts\modal_vgec.py --count 30 --corpus "D:\Project\VLSP_VGEC_2026\data\VGEC 2026_v2.json"
```

Runner tải corpus lên Modal Volume `vgec-source`, tải trọng số Qwen vào
`vgec-hf-cache`, và giữ checkpoint trong `vgec-runs`. Lệnh local lặp các vòng
sinh/kiểm, lưu checkpoint sau mỗi vòng; nếu phiên bị gián đoạn, chạy lại cùng lệnh
để tiếp tục. Khi đủ số câu yêu cầu, runner kiểm schema và tải kết quả về
`outputs/vgec-label2-modal-10/dataset.jsonl` với mặc định 10 câu, hoặc thư mục
`vgec-label2-modal-<count>` tương ứng. Có thể đổi đích bằng `--output`.
Mỗi mức `count` dùng checkpoint riêng.
Không chạy hai phiên với cùng cấu hình cùng lúc vì chúng dùng chung checkpoint.

Modal yêu cầu tài khoản và phương thức thanh toán để cấp GPU. Lần đầu build image
và tải khoảng 17,4 GB trọng số model có thể mất thời gian. Runner này chưa được
chạy trên tài khoản Modal trong phiên phát triển; kiểm thử local không chứng minh
chất lượng sinh hay bộ nhớ GPU thực tế trên Modal.

Nếu cần tải thủ công kết quả và audit từ Volume:

```powershell
.\.venv\Scripts\python.exe -m modal volume get vgec-runs /vgec-label2-modal-10/dataset.jsonl .\outputs\modal-dataset.jsonl
.\.venv\Scripts\python.exe -m modal volume get vgec-runs /vgec-label2-modal-10/audit.jsonl .\outputs\modal-audit.jsonl
```

Profile Modal ở `configs/modal.json` dùng cùng generator/verifier 4-bit và tập
loại trừ như profile local, với đường dẫn được ánh xạ sang container Modal.

## Chạy nhanh qua API

Core chỉ cần Python **3.11+**, không cần cài thư viện bên ngoài khi dùng API.
Chạy các lệnh từ thư mục dự án:

```powershell
python -m vgec doctor --config configs/api.json
python -m unittest discover -s tests -v
```

Sửa `configs/api.json` cho đúng hai endpoint tương thích Chat Completions:

- Generator: mặc định `http://localhost:8000/v1`.
- Verifier: mặc định `http://localhost:8001/v1`.
- `model` phải đúng tên model server phục vụ. Endpoint từ xa dùng HTTPS.
- Nếu API cần key, thêm `"api_key_env": "VGEC_API_KEY"` vào cấu hình model,
  rồi đặt `$env:VGEC_API_KEY = "..."` trong terminal. Không ghi key vào file JSON.
- `extra_body.chat_template_kwargs.enable_thinking=false` dành cho server hỗ trợ
  chat template Qwen. Nếu provider dùng tên tham số khác, sửa `extra_body` theo tài liệu provider.

Sau khi model server đã sẵn sàng:

```powershell
python -m vgec run --config configs/api.json
python -m vgec validate outputs/vgec-label2-api/dataset.jsonl
```

Nếu tự phục vụ trên **Linux/server GPU đủ bộ nhớ**, model card Qwen hỗ trợ vLLM.
Ví dụ khởi động ở hai terminal trên máy có đủ tài nguyên:

```bash
vllm serve Qwen/Qwen3.5-4B --port 8000 --max-model-len 8192
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8001 --max-model-len 8192
```

Các lệnh này yêu cầu đã cài bản vLLM tương thích model. Không chạy hai model
full precision đồng thời trên GPU 6 GB. Có thể dùng một server, đổi model giữa
hai bước `generate` và `verify`, đồng thời cấu hình hai vai trò dùng cùng URL.

## Chạy local với Transformers 4-bit

Máy đã kiểm tra có RTX 4050 Laptop, khoảng 6 GB VRAM. Profile local chỉ nạp một
model mỗi giai đoạn và giải phóng trước giai đoạn tiếp theo. 4-bit giảm bộ nhớ
trọng số; khả năng chạy còn phụ thuộc context, cache, phiên bản CUDA và phần
vision của Qwen3.5. Chưa đo VRAM/tốc độ thực tế của profile này trên máy của bạn.

Python trên PATH là bản MSYS2; CPython có CUDA đã tìm thấy ở venv dự án cũ
và Gen có `.venv` riêng. Khi thiết lập trên máy khác, dùng CPython có pip (hoặc Linux/WSL/Colab)
và PyTorch có CUDA phù hợp, rồi:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[local]"
python -m vgec doctor --config configs/local-4bit.json
python -m vgec run --config configs/local-4bit.json
```

Lần chạy đầu Transformers tải trọng số từ Hugging Face. Để tái lập thí nghiệm,
đổi `revision: "main"` thành commit SHA cụ thể của model trước khi bắt đầu run,
và lưu `python -m pip freeze` cùng báo cáo. Seed cố định không đảm bảo đầu ra
giống nhau tuyệt đối giữa phần cứng/server khác nhau.

## Quy trình và điều kiện nhận mẫu

1. Sinh một câu đúng tự nhiên hoặc lấy từ corpus câu đúng.
2. Chèn đúng một lỗi nhãn 2 trong một cụm; lưu span lỗi và span đúng.
3. Kiểm tra thay span khôi phục **chính xác** câu đúng; loại các thay đổi chỉ
   liên quan từ công cụ/dấu câu bằng heuristic bảo thủ. Heuristic chỉ dùng loại
   bớt ứng viên, không tự chứng minh nhãn.
4. Verifier phân loại câu lỗi theo đủ 5 nhãn, **không nhận** nhãn đích,
   câu sửa, giải thích hoặc subtype của generator.
5. Verifier phân loại riêng câu đúng, phải trả 0.
6. Verifier kiểm cặp: một lỗi, sửa tối thiểu, giữ ý nghĩa dự định, lỗi thuộc cụm,
   không có cách sửa đơn thuần bằng từ công cụ/dấu câu, lỗi có tính tự nhiên.
7. Khử trùng chính xác/gần trùng cả Input và câu sửa; phân bố lựa chọn theo
   các nhóm cấu trúc yêu cầu. Subtype là yêu cầu sinh, chưa phải nhãn phụ đã kiểm chứng.
8. Chỉ khi đủ số mẫu mới gán ID và xuất file cuối.

Confidence mặc định ≥ 0.85 ở hai bước phân loại. Đây là điểm **model tự báo**,
không phải xác suất đúng đã hiệu chuẩn. Cần đọc mẫu trong `audit.jsonl` trước
khi sử dụng dữ liệu cho thí nghiệm quan trọng. Việc kiểm nhãn độc lập giảm thiên
lệch nhưng hai model cùng họ Qwen vẫn có thể mắc lỗi tương quan.

Dedup dùng NFC + casefold + chuẩn hóa khoảng trắng và độ giống chuỗi ký tự
`SequenceMatcher` ≥ 0.9. Chuỗi xuất được giữ nguyên. Đây không phải semantic
dedup: các câu diễn đạt khác nhau nhưng cùng ý vẫn có thể lọt.

## Cấu hình dữ liệu, model bổ sung và tiến độ

Đường dẫn trong config được tính tương đối từ thư mục chứa config.

- `seed_path: null`: tự sinh câu đúng.
- `seed_path: "../data/clean.jsonl"`: mỗi dòng dạng `{"sentence":"Một câu đúng."}`.
  Có mẫu định dạng tại `examples/seeds.jsonl`; 5 câu mẫu không đủ sinh 300 bản ghi.
- `holdout_paths: ["../data/dev.jsonl", "../data/test.jsonl"]`: lọc rò rỉ.
  File nhận mảng JSON/JSONL theo schema gold VGEC. Cả câu lỗi lẫn mọi câu sửa đều được so trùng;
  ví dụ few-shot gần trùng holdout cũng bị loại khỏi prompt.
- `hard_examples_path`: file mẫu khó đã được con người kiểm tra; xem phần dưới.
- `round_size`: số lượt thử mỗi lô, không phải số mẫu chắc chắn được nhận.
- `max_rounds`: ngân sách lượt sinh; có thể tăng rồi chạy lại cùng config để resume.
- `supplementary`: mặc định tắt. Profile `configs/api-with-secondary.json` bật
  `Qwen/Qwen2.5-7B-Instruct` ở port 8002; từ vòng thứ tư, dùng xen kẽ nếu vẫn
  thiếu mẫu. Đây là cơ chế bổ sung chất lượng/số lượng, không che lỗi kết nối.

Chạy từng giai đoạn khi cần quản lý GPU/server:

```powershell
python -m vgec generate --config configs/api.json
python -m vgec verify --config configs/api.json
python -m vgec status --config configs/api.json
python -m vgec export --config configs/api.json
```

`generate` thực hiện một lô; `run` tự lặp. `export` từ chối khi chưa đủ 300 mẫu.
Chạy lại `run` sẽ tiếp tục từ checkpoint. Các lỗi API tạm thời được retry có giới
hạn; lỗi model bị cắt, sai key hoặc kết nối hỏng được báo rõ. Thay model, corpus,
prompt hoặc ngưỡng lọc phải dùng `run_dir` mới để không trộn kết quả khác cấu hình.
Nếu process bị kill cứng, chỉ xóa `.run.lock` sau khi xác nhận không còn process chạy.

## Đầu ra

```text
outputs/vgec-label2-api/
  dataset.jsonl       # 300 object, mỗi dòng một object đúng schema
  dataset.json        # Cùng dữ liệu ở dạng mảng JSON
  checkpoint.json    # Cấu hình, ứng viên, quyết định; dùng resume
  audit.jsonl        # Metadata và bằng chứng kiểm duyệt tách riêng
  report.json        # Số lượt thử, số nhận/loại và lý do
```

Ví dụ cấu trúc (không phải mẫu đã chạy qua Qwen):

```json
{"ID":120001,"Input":"Tôi mua một chiếc gỗ bàn nhỏ.","Output":{"Error_Type_Id":2,"Fixed Sentence":["Tôi mua một chiếc bàn gỗ nhỏ."]}}
```

Không thêm metadata vào `dataset.*`; không điền mẫu bị loại cho đủ số lượng.
File cuối đã tồn tại với nội dung khác sẽ không bị ghi đè.

## Đánh giá classifier trên dev và khai thác mẫu khó

Tập tăng cường chỉ có nhãn 2 phải được ghép với **train đa nhãn thật** để huấn
luyện classifier. Tập dev đánh giá nên đại diện đủ nhãn 0–4. Không đưa dev vào
train, corpus seed hoặc hard mining. ID trong mỗi file phải duy nhất.

Baseline nhẹ: TF-IDF ký tự + Logistic Regression. Cần cài dependency tùy chọn:

```powershell
python -m pip install -e ".[classifier]"
python -m vgec train-classifier --train data/train.jsonl --dev data/dev.jsonl --output-dir outputs/baseline --folds 5
python -m vgec mine --train data/train.jsonl --predictions outputs/baseline/train_oof_predictions.jsonl --holdout data/dev.jsonl --output outputs/hard-review.jsonl
```

Baseline xuất `dev_metrics.json`, `dev_predictions.jsonl` và dự đoán **out-of-fold**
trên train. Mỗi lớp train cần ít nhất `folds` mẫu. Trước khi chia fold, nên loại
các mẫu trùng và các biến thể cùng câu gốc để tránh rò rỉ giữa folds.

`mine` xếp lỗi classifier và mẫu classifier chưa chắc chắn để rà soát, giữ
nguyên nhãn gold. Mỗi dòng có `record`, `split: "train"`, `human_verified: false`.
Sau khi đọc và xác nhận ngữ pháp, chỉ giữ mẫu tốt, đổi cờ thành `true`, đặt file
vào `hard_examples_path` của **run mới**. Pipeline chỉ dùng các mẫu nhãn 2 đã xác
nhận làm few-shot, không dùng lại nguyên câu để xuất data. Bất đồng verifier của
ứng viên tổng hợp nằm trong checkpoint để kiểm tra, không tự trở thành gold.

Có thể dùng classifier khác; file dự đoán cần:

```json
{"ID":1,"label":2,"confidence":0.91}
```

## Đánh giá GEC theo metric chính thức

Chưa có tên benchmark/scorer chính thức trong yêu cầu, nên hệ thống không tự
chọn BLEU, M2 hoặc ERRANT thay thế. Adapter gọi script benchmark do bạn cung cấp,
giữ gold reference ở định dạng gốc và không tự token hóa tiếng Việt.

Tạo dự đoán GEC bằng model cấu hình sẵn hoặc nhập dự đoán model đã huấn luyện:

```powershell
python -m vgec predict --config configs/api.json --data data/dev.jsonl --role verifier --output outputs/dev-qwen.jsonl
python -m vgec evaluate --gold data/dev.jsonl --predictions outputs/dev-qwen.jsonl --output outputs/dev-classifier-metrics.json
```

Dự đoán GEC có dạng `{"ID":1,"label":2,"fixed":"Câu sửa.","confidence":0.91}`.
Nhãn 0 phải giữ Input chính xác cả khoảng trắng. Gold không được gửi vào model;
few-shot gần trùng dữ liệu đánh giá bị loại trước khi gọi model.

Macro-F1 dùng **tập nhãn cố định [0,1,2,3,4]**, `zero_division=0`, kèm precision,
recall, support từng nhãn và confusion matrix. Dev chỉ chứa nhãn 2 dù dự đoán
đúng hết cũng cho macro-F1 0.2 theo quy ước này; đó không phải đánh giá đa lớp đầy đủ.

Sao chép và sửa `configs/scorer.example.json` theo benchmark thực tế:

```powershell
python -m vgec evaluate --gold data/dev.jsonl --predictions outputs/dev-qwen.jsonl --scorer-config configs/scorer.json --output outputs/dev-full-metrics.json
```

`scorer.example.json` là hợp đồng adapter, chưa chứa scorer chạy được. `metric_key`
`F0.5` chỉ là ví dụ, phải đổi cho đúng benchmark. `command` là mảng argv, chạy
không qua shell; script adapter của bạn gọi scorer chính thức và in **một object
JSON chứa các metric dạng số** lên stdout. Log khác ghi stderr. Các placeholder:

- `{gold}`: đường dẫn `gold_reference` gốc.
- `{predictions}`: file một câu dự đoán mỗi dòng, sắp theo thứ tự `--gold` JSONL.
- `{sources}`: Input theo cùng thứ tự; `{gold_jsonl}`: dữ liệu gold VGEC.

Bạn phải bảo đảm reference gốc có cùng thứ tự/số câu với gold JSONL và adapter
dùng đúng tokenizer/config của benchmark. Khi không cấu hình scorer,
`gec_official` là `null`, không báo điểm GEC giả. Dự đoán classifier-only không
thể dùng để tính metric GEC. Adapter đã được kiểm thử với scorer fixture, chưa
chạy scorer chính thức do chưa được cung cấp.

## Kiểm thử và nguồn kỹ thuật

```powershell
python -m unittest discover -s tests -v
python -m compileall -q vgec tests
```

Tests dùng dữ liệu/HTTP server giả lập, không tải model, không gọi API tính phí.
Chúng kiểm schema/ID 300 mẫu, lỗi JSON, nhãn 0, khử trùng, cô lập dev, verifier
mù, resume, bảo vệ output, HTTP thật trên localhost và adapter scorer bằng subprocess.
Backend ML chưa được chạy với trọng số thật trong phiên triển khai này.

- [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B): phục vụ API, tắt thinking.
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).
- [Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5): chọn lớp model theo config.
- [Bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes): cấu hình NF4.
- [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
  và [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

Đã dùng `find-skills`: CLI npm bị hạn chế kết nối; tra cứu web thấy
[huggingface-datasets](https://www.skills.sh/huggingface/skills/huggingface-datasets)
và [huggingface-community-evals](https://www.skills.sh/huggingface/skills/huggingface-community-evals)
từ Hugging Face. Chúng phục vụ đọc dữ liệu/đánh giá chung, không triển khai quy ước
VGEC này, nên không cài thêm skill vào máy.
