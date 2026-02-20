"""Run Deribit live collector as: python3 -m nexus_quant.projects.crypto_options.collectors"""
from .deribit_live_collector import run_loop, collect_once, status_report
import sys

if "--loop" in sys.argv:
    run_loop()
elif "--status" in sys.argv:
    print(status_report())
else:
    collect_once()
    print(status_report())
