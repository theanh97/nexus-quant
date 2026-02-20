# DECISION-20260220-001

- Topic: Multi-model panel để giảm bias và giảm chi phí phản biện
- Decision: `GO`

## Mục tiêu

Không để kết luận quan trọng đến từ một model/agent duy nhất. Dùng panel nhiều lớp: rẻ cho vòng rộng, mạnh cho vòng chốt.

## Metric so sánh chính

1. Input/Output cost per 1M tokens
2. Context window
3. Latency phù hợp cho loop liên tục
4. Chất lượng trên eval nội bộ (không chỉ benchmark công khai)
5. Tỷ lệ phản biện tạo ra action có giá trị

## Cost snapshot (API, tại thời điểm quyết định)

- OpenAI GPT-5: input `$1.25`, output `$10`
- OpenAI GPT-5 mini: input `$0.25`, output `$2`
- Anthropic Opus 4.1: input `$15`, output `$75`
- Anthropic Sonnet 4: input `$3`, output `$15`
- Gemini 2.5 Pro: input `$1.25` (<=200k), output `$10` (<=200k)
- Gemini 2.5 Flash: input `$0.30`, output `$2.50`
- DeepSeek reasoner: input `$0.55`, output `$2.19`

## Routing policy (v1)

- Pass 1 (wide, cheap): GPT-5 mini + Gemini 2.5 Flash + DeepSeek reasoner
- Pass 2 (dissent): model có ý kiến trái chiều nhất được ưu tiên phản biện sâu
- Pass 3 (arbiter): GPT-5 (hoặc Sonnet 4 khi cần cross-vendor)
- High-cost cap: tối đa 1 call model đắt cho mỗi decision

## Go/No-Go gates

- `GO` nếu có consensus + dissent đã xử lý + eval card đạt ngưỡng
- `HOLD` nếu groundedness thấp hoặc chi phí test cao
- `NO-GO` nếu không có cơ chế kiểm chứng thực nghiệm

## Rollback condition

- Nếu cost/decision tăng >30% trong 2 tuần mà quality không tăng có ý nghĩa, rollback về panel tối giản.
