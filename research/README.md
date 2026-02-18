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

