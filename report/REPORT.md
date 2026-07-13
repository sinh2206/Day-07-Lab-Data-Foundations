# Báo cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Cập nhật trước khi nộp]
**Nhóm:** [Cập nhật trước khi nộp]
**Ngày:** 13/07/2026

## 1. Warm-up

### Cosine similarity

Cosine similarity đo mức độ cùng hướng của hai vector embedding. Điểm gần 1 cho thấy hai đoạn văn bản có nghĩa gần nhau; điểm gần 0 cho thấy chúng ít liên quan.

- HIGH: "Python được dùng cho phân tích dữ liệu." và "Python hỗ trợ các quy trình xử lý dữ liệu." Cả hai đều nói về ứng dụng Python trong data.
- LOW: "Python hỗ trợ các quy trình xử lý dữ liệu." và "Người dùng cần đặt lại mật khẩu tại trang đăng nhập." Hai câu thuộc hai chủ đề khác nhau.

Cosine similarity phù hợp với text embedding vì quan tâm hướng ngữ nghĩa của vector hơn độ lớn. Euclidean distance dễ bị ảnh hưởng bởi độ dài vector, trong khi độ dài văn bản không phản ánh trực tiếp mức độ tương đồng ngữ nghĩa.

### Chunking math

Với 10.000 ký tự, `chunk_size=500`, `overlap=50`:

`ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 23` chunks.

Nếu overlap tăng lên 100: `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25` chunks. Số chunk tăng vì bước trượt nhỏ hơn; đổi lại, ngữ cảnh ở ranh giới chunk được giữ tốt hơn.

## 2. Document selection

**Domain:** Tài liệu nền tảng cho retrieval và trợ lý tri thức nội bộ.

Bộ dữ liệu gồm tài liệu về Python, vector store, RAG, support và ghi chú retrieval. Chủ đề này phù hợp để so sánh chunking vì có cả đoạn kỹ thuật dài, quy trình và nội dung tiếng Việt.

| # | Tài liệu | Nguồn | Kích thước (byte) | Metadata đề xuất |
|---|---|---|---:|---|
| 1 | `python_intro.txt` | `data/` | 1.953 | `source`, `extension`, `language=en`, `category=python` |
| 2 | `vector_store_notes.md` | `data/` | 2.149 | `source`, `extension`, `language=en`, `category=vector-store` |
| 3 | `rag_system_design.md` | `data/` | 2.416 | `source`, `extension`, `language=en`, `category=rag` |
| 4 | `customer_support_playbook.txt` | `data/` | 1.703 | `source`, `extension`, `language=en`, `category=support` |
| 5 | `chunking_experiment_report.md` | `data/` | 2.008 | `source`, `extension`, `language=en`, `category=chunking` |
| 6 | `vi_retrieval_notes.md` | `data/` | 2.188 | `source`, `extension`, `language=vi`, `category=retrieval` |

`source` giúp truy vết tài liệu gốc; `language` và `category` là hai trường phù hợp để lọc kết quả trước khi xếp hạng. `extension` giúp phân biệt định dạng khi bộ dữ liệu được mở rộng.

## 3. Chunking strategy

**Strategy của tôi:** `RecursiveChunker(chunk_size=500)`.

Chunker ưu tiên tách theo đoạn trống (`\n\n`), sau đó là dòng, câu, khoảng trắng và cuối cùng là ký tự. Mỗi phần vượt giới hạn sẽ được tách tiếp ở mức phân cách nhỏ hơn. Cách này giữ được đoạn văn và câu trọn vẹn nếu có thể, nhưng vẫn đảm bảo chunk không vượt quá giới hạn trong trường hợp văn bản dài hoặc không có cấu trúc.

Strategy này phù hợp với bộ tài liệu Markdown/TXT hơn fixed-size vì tài liệu có heading và paragraph rõ ràng. Sentence chunking dễ đọc nhưng không kiểm soát được kích thước khi câu dài; fixed-size ổn định về kích thước nhưng có thể cắt giữa ý.

| Strategy | Đặc điểm | Kỳ vọng retrieval |
|---|---|---|
| `fixed_size` | Cửa sổ ký tự có overlap | Ổn định, nhưng có nguy cơ cắt ý |
| `by_sentences` | Gom tối đa 3 câu | Dễ đọc, phụ thuộc độ dài câu |
| `recursive` | Ưu tiên cấu trúc lớn rồi mới tách nhỏ | Cân bằng coherence và giới hạn kích thước |

Số chunk và độ dài trung bình phải lấy từ `ChunkingStrategyComparator().compare()` khi chạy trên cùng một tài liệu; không ghi số liệu đoán trong báo cáo.

## 4. My approach

`SentenceChunker` dùng regex `(?<=[.!?])\s+` để tách sau dấu kết thúc câu, bỏ khoảng trắng thừa và gom các câu theo `max_sentences_per_chunk`. Văn bản rỗng trả về danh sách rỗng; một giá trị giới hạn nhỏ hơn 1 được chuyển thành 1.

`RecursiveChunker` dùng đệ quy: nếu đoạn đã đủ ngắn thì trả về ngay; nếu không, nó thử separator hiện tại và chỉ đệ quy các mảnh vẫn còn quá dài bằng separator tiếp theo. Nếu hết separator, nó cắt theo `chunk_size`, nên thuật toán luôn kết thúc.

`compute_similarity` áp dụng `dot(a, b) / (||a|| * ||b||)` và trả `0.0` khi một vector có độ lớn bằng 0. `ChunkingStrategyComparator` chạy cả ba chunker và trả `chunks`, `count`, `avg_length` để so sánh.

`EmbeddingStore` tạo record gồm `id`, `content`, `metadata`, `embedding`; metadata được bổ sung `doc_id`. Chế độ mặc định lưu trong bộ nhớ. Search embed query, tính dot product và sắp xếp giảm dần; với embedding đã normalize, dot product tương đương cosine similarity. `search_with_filter` lọc metadata trước khi xếp hạng; `delete_document` xóa tất cả record có cùng `doc_id`.

`KnowledgeBaseAgent` lấy top-k chunks, đánh số context và chèn chúng vào prompt. Prompt yêu cầu LLM chỉ trả lời theo context, hoặc báo thiếu thông tin nếu retrieval không đủ.

### Test results

Chưa chạy theo yêu cầu của sinh viên. Cần dán output của `pytest tests/ -v` và số test pass vào đây sau khi kiểm tra.

## 5. Similarity predictions

| # | Sentence A | Sentence B | Dự đoán |
|---|---|---|---|
| 1 | Python dùng cho data analysis. | Python hỗ trợ xử lý dữ liệu. | Cao |
| 2 | Vector store lưu embeddings. | Cơ sở dữ liệu vector lưu vector để tìm kiếm. | Cao |
| 3 | RAG đưa context vào prompt. | Chunking chia tài liệu thành đoạn nhỏ. | Trung bình |
| 4 | Đặt lại mật khẩu tại trang đăng nhập. | Triển khai API billing. | Thấp |
| 5 | Metadata filter tăng precision. | Lọc theo ngôn ngữ làm giảm nhiễu. | Cao |

Cần ghi thêm điểm thực tế từ cùng một embedding backend. `_mock_embed` là pseudo-random deterministic để test tính đúng của code, không nên dùng điểm của nó để kết luận về tương đồng ngữ nghĩa; nên dùng `LocalEmbedder` hoặc backend embedding thật khi làm phần này.

## 6. Benchmark queries và gold answers

| # | Query | Gold answer | Metadata filter gợi ý |
|---|---|---|---|
| 1 | Python thường được dùng cho những công việc nào? | Automation, backend, data analysis, scientific computing và machine learning. | `category=python` |
| 2 | Pipeline vector search có các bước nào? | Chunk tài liệu, embed chunk, lưu vector và metadata, embed query rồi rank theo similarity. | `category=vector-store` |
| 3 | RAG giảm hallucination như thế nào? | Retrieve tài liệu trước, chèn evidence vào prompt và báo thiếu thông tin khi context yếu. | `category=rag` |
| 4 | Metadata nào cần tách trong tài liệu support? | Customer-facing, support-only và engineering-only để tránh lộ thông tin hoặc lấy sai ngữ cảnh. | `category=support` |
| 5 | Vì sao nên lọc theo ngôn ngữ trong retrieval? | Để tránh lấy tài liệu khác ngôn ngữ và tăng precision cho truy vấn có phạm vi. | `language=vi` |

Kết quả top-3, score và tóm tắt agent answer cần được ghi sau khi chạy cùng một backend và cùng bộ 5 query. Không tự điền score hoặc đánh giá relevant khi chưa có kết quả chạy.

## 7. Failure analysis và bài học

Failure case cần kiểm tra: truy vấn tiếng Việt trên toàn bộ tài liệu Anh có thể retrieve nhiều chunk nhiều từ khóa nhưng không đúng ngữ cảnh. Nguyên nhân có thể là embedding đa ngôn ngữ, query mơ hồ hoặc metadata `language` chưa được áp dụng. Cách cải thiện là gắn metadata đầy đủ, thử `search_with_filter(..., {"language": "vi"})`, và thêm tài liệu tiếng Việt cùng chủ đề.

Bài học chính là retrieval phụ thuộc vào dữ liệu và chunk boundary, không chỉ phụ thuộc vector store. Cần bổ sung phần so sánh với thành viên khác và bài học từ demo sau khi nhóm thực hiện hoạt động đó; không thể trung thực tự tạo nội dung này khi chưa có thông tin nhóm.

## Tự đánh giá

| Tiêu chí | Trạng thái |
|---|---|
| Core implementation | Đã hoàn thành trong `src/`; cần chạy test để xác nhận |
| Warm-up và My approach | Đã ghi |
| Document selection và benchmark queries | Đã chuẩn bị từ 6 tài liệu mẫu |
| Similarity actual scores | Cần chạy backend embedding |
| Retrieval results, so sánh nhóm, demo | Cần thực hiện và ghi nhận thực tế |
