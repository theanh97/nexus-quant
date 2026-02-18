# NEXUS Quant Runbook (v1)

Mục tiêu của repo này: chạy R&D định lượng theo kiểu **audit được + tái lập + có gate**, giống tinh thần NEXUS gốc nhưng chuyển trọng tâm sang Quant/Trading.

---

## 1) Data Contract & Anti-bias

### Hard rules (không thỏa là không chạy)
- Timeline phải **sorted + unique + strictly increasing**.
- `close > 0` cho mọi bar.
- Series length phải khớp timeline.
- Không được dùng thông tin tương lai để tạo signal (no look-ahead).

### Những bias phải né
- Look-ahead bias (dùng dữ liệu của tương lai).
- Universe leakage (chọn top volume dựa trên volume tương lai).
- Survivorship bias (bỏ coin delist/đổ về chỉ coin còn sống).
- Time alignment lỗi (timestamp lệch timezone / funding event lệch bar).

### Implementation
- Data quality gate: `nexus_quant/data/quality.py`
- Mỗi run ghi: `artifacts/runs/<run_id>/data_quality.json`
- Nếu fail gate: CLI sẽ dừng sớm.

---

## 2) Backtest Assumptions (rõ ràng để audit)

### Timeline & execution
- Mỗi bước `idx` tương ứng 1 bar kết thúc tại `timeline[idx]`.
- Price PnL dùng **weights đã hold** trong bar `(idx-1 -> idx)`.
- Rebalance (tính turnover + cost) diễn ra tại `timeline[idx]` (sau khi bar đóng).
- Funding event tại `timeline[idx]` áp lên vị thế tại timestamp đó (sau rebalance).

### Costs
- Model hiện tại: cost theo notional traded (nhưng cấu trúc rõ để audit)
  - fee: maker/taker (`maker_fee_rate`, `taker_fee_rate`) + `execution.style`
  - slippage: `execution.slippage_bps` (hoặc legacy `costs.slippage_rate`)
  - spread: `execution.spread_bps` (taker bị tính ~half-spread)
  - impact (optional): `execution.impact` (vd `sqrt`)
  - stress test: `costs.cost_multiplier` (self-learn stress gate sẽ x2)
- Công thức tổng quát (first-order):
  - `cost = equity * turnover * (fee_rate + slippage_rate + spread_rate + impact_rate) * cost_multiplier`
- Đây vẫn là mô hình tối thiểu để chống “alpha ảo”; khi chuyển sang data thật bạn thay bằng fee tier cụ thể + impact/slippage model phù hợp venue.

---

## 3) Benchmark v1 (Locked)

### 3 tầng
- T1: Ablation (baseline vs +NEXUS)
- T2: Public baselines (B&H BTC, equal-weight B&H)
- T3: Robustness (walk-forward windows + stress costs)

### Metrics tối thiểu
- Total return, CAGR, Vol
- Sharpe, Sortino
- Max drawdown, Calmar
- Turnover
- Beta/Corr vs BTC
- Walk-forward stability (fraction profitable, fraction MDD ok)

### Verdict (pass/fail)
- Verdict được ghi trong `metrics.json` theo các threshold trong config `risk`.

---

## 4) Verified Self-Learning (L1)

Định nghĩa: agent được phép **đề xuất** thay đổi params, nhưng chỉ được **accept** nếu:
- Pass risk gates
- Improve trên holdout theo objective (ví dụ `median_calmar`) vượt `min_uplift`
- Evidence được ghi vào ledger (append-only)

Implementation: `nexus_quant/self_learn/search.py`

### Priors (tích lũy theo thời gian)
- Khi accept candidate, hệ thống update `artifacts/memory/priors.json`
- Candidate sampling sẽ mix:
  - exploit theo priors (weighted theo lịch sử accept)
  - explore ngẫu nhiên (để tránh kẹt local optimum)
- Flags:
  - `self_learn.priors_enabled` (default true)
  - `self_learn.prior_exploit_prob` (default 0.7)

### Stress gate (khuyến nghị bật mặc định)
- Sau khi một candidate được accept trên holdout, hệ thống chạy lại **stress test phí** (mặc định `cost_multiplier x2`).
- Candidate chỉ được giữ trạng thái accept nếu pass stress gate (min uplift trên holdout khi cost tăng).

### Ablation artifact (bằng chứng để audit/PR)
Khi accept, hệ thống sinh:
- `artifacts/memory/ablations/ablation.<ts>.json`
- `artifacts/memory/ablations/ablation.<ts>.md`

---

## 5) Roles (mapping từ NEXUS gốc sang Quant)

- ORION (PM/Commander): chốt objective, đánh đổi, quyết định go/no-go.
- ATLAS (Quant R&D): generate hypotheses / strategy families / param search.
- ECHO (QA/Eval): data gate, benchmark locked, regression checks.
- CIPHER (Risk): risk gates + stress scenarios.
- FLUX (Ops/DevOps): artifact/ledger hygiene, scheduling.
- GUARDIAN (Monitor): drift alerts, stuck loop recovery, rollback protocol.

Ở v1 repo này mới có “engine” + “loop”, phần multi-agent orchestration sẽ triển khai tiếp theo (LLM router + task bus).

---

## 6) Orion Autopilot (24/7 skeleton)

Orion tạo task + chạy tuần tự:
1. `run` (baseline backtest + benchmark + ledger)
2. `improve` (self-learning + holdout + stress + ablation artifacts)
3. `wisdom` (curate checkpoint: ledger + memory -> wisdom artifacts)
4. `handoff` (tạo file bàn giao + câu hỏi cho human)

Commands:
- `python3 -m nexus_quant autopilot --config <cfg.json> --bootstrap --steps 10`
- 24/7 mode: `python3 -m nexus_quant autopilot --config <cfg.json> --bootstrap --loop --interval-seconds 300`

Monitoring:
- Heartbeat: `artifacts/state/orion_heartbeat.json`
- Guardian check: `python3 -m nexus_quant guardian --artifacts artifacts --stale-seconds 900`

Promotion:
- Xem candidate accept: `artifacts/memory/best_params.json` + `artifacts/memory/ablation_latest.json`
- Dry-run promote: `python3 -m nexus_quant promote --config <cfg.json> --best artifacts/memory/best_params.json`
- Apply promote: thêm `--apply` (tạo backup ở `artifacts/config_backups/` và ghi decision vào memory)

---

## 6.5) Wisdom Checkpoints (Long-horizon)

Mục tiêu: NEXUS tích lũy “trí nhớ dự án” theo thời gian (3 tháng/6 tháng/1 năm) theo dạng artifact có provenance.

- CLI:
  - `python3 -m nexus_quant wisdom --artifacts artifacts --tail-events 200`
- Output:
  - `artifacts/wisdom/checkpoints/wisdom.<ts>.json`
  - `artifacts/wisdom/latest.md`

Wisdom checkpoint là input an toàn cho multi-agent + LLM router sau này (không cần paste raw logs).

---

## 7) Long-Term Memory + Feedback Loop

Memory (SQLite) dùng để lưu:
- feedback từ user
- quyết định promote/reject
- notes về data/assumptions/risk
- nguồn research (provenance)

Commands:
- add: `python3 -m nexus_quant memory add --kind feedback --tags user --content "..." --meta '{}'`
- search: `python3 -m nexus_quant memory search --query "funding" --limit 20`

---

## 8) Verified Research Ingestion (Local-first)

Mục tiêu: ingest nguồn research có hash + provenance vào memory (không crawl web mặc định).

- Inbox: `research/inbox/`
- Ingest:
  - `python3 -m nexus_quant research ingest --path research/inbox --artifacts artifacts --kind source --tags research --move-to research/library`
