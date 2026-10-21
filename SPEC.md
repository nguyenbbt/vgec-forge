# VGEC data factory

## Mục tiêu và phạm vi
CLI Python 3.11+ sinh dữ liệu tiếng Việt bằng Qwen theo quy ước người dùng.
Mặc định chỉ xuất nhãn 2, đúng 300 ID liên tiếp 120001–120300. Metadata
kiểm duyệt nằm riêng, không lọt vào bản ghi giao nộp.

## Các phần và thứ tự xây dựng
1. `data`: schema nghiêm ngặt, JSONL UTF-8, khử trùng và kiểm tra đầu ra.
2. `models` + `prompts`: API chat tương thích OpenAI / Transformers, rubric,
   few-shot có điều kiện nhãn, sinh câu đúng rồi chèn lỗi ngược.
3. `pipeline`: checkpoint, xác minh nhãn mù, đánh giá phép sửa, chọn đủ số lượng.
4. `evaluation`: macro-F1 theo ID, adapter scorer GEC do benchmark cung cấp,
   baseline classifier và khai thác lỗi classifier trên train.

## Giả định đã công khai
- CLI và file cấu hình JSON; không cần giao diện web.
- API là cấu hình mặc định, local 4-bit là lựa chọn cho GPU ít VRAM.
- Chưa có corpus/dev/scorer: nhận qua đường dẫn, không giả định metric chính thức.
- Nhãn 4 chỉ dùng làm nhãn loại trừ: lỗi sửa được bằng dấu câu đơn thuần.
- Seed sinh tự động hoặc corpus câu đúng của người dùng; dev chỉ dùng lọc rò rỉ
  và đánh giá, không làm few-shot/hard mining.

## Hợp đồng
`{"ID":120001,"Input":"...","Output":{"Error_Type_Id":2,"Fixed Sentence":["..."]}}`
Không ép kiểu string/bool thành integer. Không thêm khóa vào output.
Một câu sửa không rỗng, khác Input với nhãn 2; nhãn 0 giữ từng ký tự khi đánh giá.
Thiếu số mẫu đạt chuẩn thì lỗi rõ ràng, giữ checkpoint, không xuất thiếu hoặc đệm mẫu.
Verifier khác generator; phân loại Input không nhận nhãn đích/câu sửa/rationale.
Sau đó phân loại câu đúng và kiểm tra minimal repair, một lỗi, phạm vi cụm,
không sửa được chỉ bằng từ công cụ/dấu câu, giữ ý nghĩa và tính tự nhiên.
Từ chối output model sai schema, malformed JSON và đáp án cắt cụt.
Dedup chuẩn hóa Unicode/space/case để so sánh nhưng không sửa chuỗi được xuất.

## Chất lượng và giới hạn
Confidence là điểm tự báo của model, không phải xác suất đã hiệu chuẩn.
Không khẳng định model verification thay thế thẩm định ngôn ngữ của con người.
Hard mining dùng train đã có nhãn đáng tin + dự đoán classifier theo ID;
mẫu bất đồng không tự đổi nhãn, không tự nâng thành gold.
Macro-F1 mặc định tính đủ 5 nhãn 0..4, kèm support/matrix; training từ một nhãn bị từ chối.
Metric GEC chỉ lấy từ scorer được người dùng cấu hình, giữ nguyên tokenization/reference.

## Cấu trúc và lệnh
`vgec/`: code; `tests/`: unittest; `configs/`: cấu hình; `examples/`: mẫu đầu vào.
Test: `python -m unittest discover -s tests -v`
Kiểm cú pháp: `python -m compileall -q vgec tests`
Chạy: `python -m vgec run --config configs/api.json`
Style: stdlib, type hints, hàm nhỏ, JSON serialize qua json.dumps, subprocess shell=False.

## Ranh giới
Luôn kiểm tra dữ liệu/model output, chạy tests; API key đọc từ biến môi trường.
Không tải model hoặc gọi API tính phí trong kiểm thử. Không lưu key trong checkpoint.
Không dùng eval/exec cho model output; lệnh metric chỉ từ cấu hình người vận hành.
Không ghi đè output có sẵn hoặc thay config của run đang dở một cách âm thầm.
