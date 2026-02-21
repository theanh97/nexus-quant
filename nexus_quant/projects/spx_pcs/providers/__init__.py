"""
SPX PCS Data Providers
=======================

Data được đọc trực tiếp bởi algoxpert engine (SPXW 1-min parquet + VIX 1-min).
NEXUS không access raw data — chỉ đọc processed artifacts.

Data layout trong algoxpert:
  data/spxw_YYYY/YYYYMMDD.parquet   — SPX weekly options 1-min OHLCV
  data/vix/vix_ohlc_1min.parquet    — VIX 1-minute OHLC
  data/incoming_zips/               — staging folder cho zip ingest

Để download data trên máy mới:
  cd algoxpert-3rd-alpha-spx
  ./tools/bootstrap_data.sh
"""
