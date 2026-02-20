# NEXUS Agentic R&D

Mục tiêu: biến nghiên cứu thành một pipeline có thể lặp vô hạn, tự phản biện, tự đo chất lượng, và chỉ hành động khi có bằng chứng.

## Nguyên tắc vận hành

1. Không dựa vào 1 model duy nhất cho kết luận quan trọng.
2. Mọi kết luận phải có citation và thẻ đánh giá hành động.
3. Mọi thay đổi phải đi qua decision log (go/no-go + rollback condition).
4. Tách rõ: ý tưởng → phản biện → đánh giá → quyết định.
5. Ưu tiên chi phí thấp cho vòng rộng, chỉ dùng model đắt ở vòng chốt.

## Luồng chuẩn

1. `01_intake`: ingest research mới mỗi ngày.
2. `02_hypotheses`: tạo giả thuyết testable.
3. `03_debate_logs`: panel đa-model phản biện chéo.
4. `04_eval_cards`: chấm điểm groundedness, novelty, expected value.
5. `05_decisions`: quyết định triển khai/hoãn/huỷ.
6. `07_daily_briefs`: tổng hợp ngày/tuần để cập nhật memory.
