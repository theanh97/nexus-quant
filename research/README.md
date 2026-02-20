# Research Inbox (Local-First)

Mục tiêu: giúp NEXUS "tự học có kiểm định" bằng cách **ingest nguồn research có provenance** vào long-term memory (SQLite) kèm hash.

Repo này không tự crawl internet (để tránh dữ liệu bẩn / không kiểm chứng). Thay vào đó:
- Bạn tải/copy tài liệu vào `research/inbox/`
- Chạy lệnh ingest để đưa vào memory kèm `sha256` + metadata
- (Tuỳ chọn) move vào `research/library/` để bạn curate dần theo thời gian

## Ingest

```bash
python3 -m nexus_quant research ingest --path research/inbox --artifacts artifacts --kind source --tags research --move-to research/library
```

Notes:
- Text files (`.md/.txt/.json/...`) sẽ được lưu nội dung (giới hạn dung lượng).
- Binary files (vd `.pdf`) sẽ lưu provenance + hash; nội dung không parse (stdlib-only).

## Agentic R&D Workspace

Để lưu trữ phản biện đa-model, ghi nhớ nghiên cứu, và quyết định triển khai liên tục:

- `research/agentic_rnd/01_intake/` — input research thô đã chuẩn hoá
- `research/agentic_rnd/02_hypotheses/` — giả thuyết cần test
- `research/agentic_rnd/03_debate_logs/` — log phản biện đa-agent/model
- `research/agentic_rnd/04_eval_cards/` — thẻ đánh giá chất lượng + giá trị hành động
- `research/agentic_rnd/05_decisions/` — decision log (go/no-go)
- `research/agentic_rnd/06_model_panel/` — registry model và routing policy
- `research/agentic_rnd/07_daily_briefs/` — daily brief + weekly synthesis
- `research/agentic_rnd/templates/` — template chuẩn cho vận hành tự động
