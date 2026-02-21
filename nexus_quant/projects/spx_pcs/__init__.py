"""
NEXUS Project 4: SPX Put Credit Spread (SPX PCS)
=================================================

Bridge adapter cho algoxpert-3rd-alpha-spx engine.

Strategy: Bán Put Credit Spread trên SPX (S&P 500 index options).
  - 0DTE / short-dated options
  - Delta selection: target_delta = 0.15-0.20
  - VIX gate: không trade khi VIX > 30
  - TP/SL exit: TP at 50% credit, SL at 2x credit
  - Fill model: bid_ask (conservative)

Engine: Python + Rust (spx_core via maturin)
Data: SPXW 1-minute parquet (2020-2025) + VIX 1-minute
Venue: CBOE, fee $0.65/contract

Integration mode: BRIDGE ADAPTER
  - algoxpert engine chạy độc lập
  - NEXUS đọc artifacts từ algoxpert qua SPXArtifactReader
  - NEXUS có thể trigger algoxpert qua SPXRunner (subprocess)
  - Path: env var ALGOXPERT_DIR (default: ../algoxpert-3rd-alpha-spx)

Status: development
Target Sharpe: > 0.8 (WF validated)
"""

from __future__ import annotations

PROJECT_NAME = "spx_pcs"
PROJECT_VERSION = "1.0.0"
ASSET_CLASS = "put_credit_spread"
MARKET = "equity"
STATUS = "development"
TARGET_SHARPE = 0.8
