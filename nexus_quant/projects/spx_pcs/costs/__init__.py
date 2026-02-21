"""
SPX PCS Cost Model (CBOE Options)
==================================

Fee structure for SPX options at CBOE:
  - Per-contract fee: $0.65 (algoxpert default)
  - Exchange fees typically: $0.20-0.65/contract
  - Bid-ask spread: modeled via fill_model="bid_ask" in algoxpert

Note: SPX options are European-style, cash-settled.
      No early assignment risk. Fees are flat per contract.
      1 contract = 100x multiplier.
"""

CBOE_FEE_PER_CONTRACT_USD = 0.65
CONTRACT_MULTIPLIER = 100  # 1 contract = 100 shares equivalent
