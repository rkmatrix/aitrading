# Box Trading Bot - Critical Fixes Instructions

## Overview
After comprehensive code review, identified **6 Critical** and **6 High** severity issues that must be fixed before live trading.

---

## CRITICAL FIXES TO APPLY

### Fix C1: Add Real Broker Order Execution

**Location:** `runner/box_trading_runner.py`

#### 1. Add helper methods after `__init__` method (around line 180):

```python
def _get_account_equity(self) -> float:
    """Get current account equity with caching"""
    # Cache equity for 60 seconds to avoid rate limiting
    if not hasattr(self, '_equity_cache'):
        self._equity_cache = {'value': 10000.0, 'timestamp': 0}
    
    current_time = time.time()
    if current_time - self._equity_cache['timestamp'] > 60:
        try:
            if self.broker:
                account = self.broker.client.get_account()
                self._equity_cache['value'] = float(account.equity)
                self._equity_cache['timestamp'] = current_time
                logger.debug(f"Account equity refreshed: ${self._equity_cache['value']:.2f}")
        except Exception as e:
            logger.error(f"Failed to get account equity: {e}")
    
    return self._equity_cache['value']

def _wait_for_fill(self, order_id: str, timeout_seconds: int = 30) -> Optional[Dict]:
    """Wait for order to fill with timeout"""
    if not self.broker:
        return None
    
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            status = self.broker.get_order_status(order_id)
            if status:
                order_status = status.get('status', '').lower()
                if order_status in ['filled', 'partially_filled']:
                    return status
                elif order_status in ['cancelled', 'expired', 'rejected']:
                    logger.error(f"Order {order_id} failed with status: {order_status}")
                    return None
        except Exception as e:
            logger.error(f"Error checking order status: {e}")
        
        time.sleep(0.5)
    
    logger.warning(f"Order {order_id} fill timeout after {timeout_seconds}s")
    return None

def _can_open_position_for_symbol(self, symbol: str) -> bool:
    """Check if we can open position considering correlation groups"""
    correlation_groups = self.config.get("correlation_groups", {})
    max_correlated = self.config.get("max_correlated_positions", 1)
    
    # Find which group this symbol belongs to
    symbol_group = None
    for group_name, symbols in correlation_groups.items():
        if symbol in symbols:
            symbol_group = group_name
            break
    
    # If symbol not in any group, allow it
    if not symbol_group:
        return True
    
    # Count how many positions we have in this group
    group_symbols = correlation_groups[symbol_group]
    positions_in_group = sum(1 for sym in self.positions if sym in group_symbols)
    
    return positions_in_group < max_correlated
```

#### 2. Replace `_calculate_position_size` method (around line 501):

```python
def _calculate_position_size(self, signal: TradeSignal) -> int:
    """Calculate position size based on risk"""
    # Get real account equity
    account_equity = self._get_account_equity()
    
    # Get risk per trade
    base_risk = self.config.get("base_risk_per_trade", 0.02)
    max_risk = self.config.get("max_risk_per_trade", 0.04)
    min_risk = self.config.get("min_risk_per_trade", 0.01)
    
    # Adjust risk based on confidence
    if signal.confidence >= 0.90:
        risk_percent = max_risk
    elif signal.confidence >= 0.80:
        risk_percent = (base_risk + max_risk) / 2
    elif signal.confidence >= 0.75:
        risk_percent = base_risk
    else:
        risk_percent = min_risk
    
    # Calculate dollar risk
    risk_amount = account_equity * risk_percent
    
    # Calculate position size based on stop distance
    stop_distance = abs(signal.current_price - signal.stop_loss)
    
    if stop_distance <= 0:
        return 0
    
    shares = int(risk_amount / stop_distance)
    
    # Ensure at least 1 share
    return max(1, shares)
```

#### 3. Replace `_execute_entry` method (around line 460):

```python
def _execute_entry(self, signal: TradeSignal, current_time: datetime):
    """Execute trade entry with real broker orders"""
    try:
        # Check correlation groups
        if not self._can_open_position_for_symbol(signal.symbol):
            logger.info(f"Skipping {signal.symbol} - correlation group limit reached")
            return
        
        # Calculate position size
        position_size = self._calculate_position_size(signal)
        
        if position_size <= 0:
            logger.warning(f"Position size calculation returned 0 for {signal.symbol}")
            return
        
        # Execute real broker order
        if not self.broker:
            logger.error("No broker available - cannot execute trade")
            return
        
        order = {
            "symbol": signal.symbol,
            "side": signal.action.lower(),  # "buy" or "sell"
            "qty": position_size,
            "order_type": "market"
        }
        
        logger.info(f"Submitting {signal.action} order for {signal.symbol}: "
                   f"{position_size} shares @ ${signal.current_price:.2f}")
        
        # Submit order to broker
        resp = self.broker.submit_order(order)
        
        # Check for errors
        if isinstance(resp, dict):
            if resp.get("error") or resp.get("order_submitted") == False:
                logger.error(f"Order failed for {signal.symbol}: {resp.get('error', 'Unknown error')}")
                return
        
        # Get order ID
        order_id = getattr(resp, 'id', None) if resp else None
        if not order_id:
            logger.error(f"No order ID received for {signal.symbol}")
            return
        
        # Wait for fill confirmation
        fill_status = self._wait_for_fill(order_id, timeout_seconds=30)
        if not fill_status:
            logger.error(f"Order {order_id} did not fill for {signal.symbol}")
            return
        
        # Get actual fill price
        filled_price = float(fill_status.get('filled_avg_price', signal.current_price))
        filled_qty = int(float(fill_status.get('filled_qty', position_size)))
        
        logger.info(f"✅ Order filled: {filled_qty} shares @ ${filled_price:.2f}")
        
        # Create position AFTER broker confirms
        position = Position(
            symbol=signal.symbol,
            action=signal.action,
            entry_price=filled_price,
            quantity=filled_qty,
            stop_loss=signal.stop_loss,
            take_profit_targets=signal.take_profit_targets,
            entry_time=current_time,
            signal=signal
        )
        
        self.positions[signal.symbol] = position
        
        # Update stats
        self.daily_stats["trades"] += 1
        
        # Send alert
        self._send_trade_entry_alert(position)
        
        logger.info(f"Position opened: {position}")
        
    except Exception as e:
        logger.error(f"Error executing entry for {signal.symbol}: {e}", exc_info=True)
```

#### 4. Replace `_partial_exit` method (around line 610):

```python
def _partial_exit(self, position: Position, exit_price: float, exit_quantity: int, reason: str):
    """Execute partial exit with real broker order"""
    if exit_quantity <= 0:
        return False
    
    try:
        # Execute real broker order
        if self.broker:
            order = {
                "symbol": position.symbol,
                "side": "sell" if position.action == "BUY" else "buy",
                "qty": exit_quantity,
                "order_type": "market"
            }
            
            resp = self.broker.submit_order(order)
            
            # Check for errors
            if isinstance(resp, dict) and (resp.get("error") or resp.get("order_submitted") == False):
                logger.error(f"Partial exit order failed for {position.symbol}: {resp.get('error')}")
                return False
            
            # Wait for fill
            order_id = getattr(resp, 'id', None) if resp else None
            if order_id:
                fill_status = self._wait_for_fill(order_id, timeout_seconds=15)
                if fill_status:
                    exit_price = float(fill_status.get('filled_avg_price', exit_price))
                    exit_quantity = int(float(fill_status.get('filled_qty', exit_quantity)))
        
        # Calculate P&L for this portion
        if position.action == "BUY":
            pnl = (exit_price - position.entry_price) * exit_quantity
        else:
            pnl = (position.entry_price - exit_price) * exit_quantity
        
        position.realized_pnl += pnl
        position.remaining_quantity -= exit_quantity
        
        position.partial_exits.append({
            "price": exit_price,
            "quantity": exit_quantity,
            "pnl": pnl,
            "reason": reason,
            "time": datetime.now(ZoneInfo("America/New_York"))
        })
        
        logger.info(f"Partial exit for {position.symbol}: {exit_quantity} shares @ ${exit_price:.2f}, "
                   f"PnL=${pnl:.2f} ({reason})")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in partial exit for {position.symbol}: {e}")
        return False
```

#### 5. Replace `_close_position` method (around line 637):

```python
def _close_position(self, symbol: str, exit_price: float, reason: str, current_time: datetime) -> bool:
    """Close position with real broker order"""
    position = self.positions.get(symbol)
    
    if not position:
        return False
    
    try:
        # Execute real broker close order
        if self.broker:
            order = {
                "symbol": position.symbol,
                "side": "sell" if position.action == "BUY" else "buy",
                "qty": position.remaining_quantity,
                "order_type": "market"
            }
            
            resp = self.broker.submit_order(order)
            
            # Check for errors
            if isinstance(resp, dict) and (resp.get("error") or resp.get("order_submitted") == False):
                logger.error(f"Close order failed for {position.symbol}: {resp.get('error')}")
                return False
            
            # Wait for fill
            order_id = getattr(resp, 'id', None) if resp else None
            if order_id:
                fill_status = self._wait_for_fill(order_id, timeout_seconds=15)
                if fill_status:
                    exit_price = float(fill_status.get('filled_avg_price', exit_price))
                else:
                    logger.error(f"Close order did not fill for {symbol}")
                    return False
        
        # Calculate final P&L
        if position.action == "BUY":
            pnl = (exit_price - position.entry_price) * position.remaining_quantity
        else:
            pnl = (position.entry_price - exit_price) * position.remaining_quantity
        
        position.realized_pnl += pnl
        total_pnl = position.realized_pnl
        
        # Update daily stats
        self.daily_stats["total_pnl"] += total_pnl
        if total_pnl > 0:
            self.daily_stats["wins"] += 1
            self.consecutive_losses = 0
        else:
            self.daily_stats["losses"] += 1
            self.consecutive_losses += 1
        
        # Send alert
        self._send_trade_exit_alert(position, exit_price, reason, total_pnl)
        
        # Update strategy stats
        if hasattr(self.strategy, 'record_stop_out') and reason == "Stop loss":
            self.strategy.record_stop_out(symbol, current_time)
        
        if hasattr(self.strategy, 'update_performance'):
            self.strategy.update_performance(symbol, total_pnl, current_time)
        
        logger.info(f"Position closed: {symbol} @ ${exit_price:.2f}, Total PnL=${total_pnl:.2f} ({reason})")
        
        # Remove from tracking ONLY after successful broker close
        return True
        
    except Exception as e:
        logger.error(f"Error closing position for {symbol}: {e}", exc_info=True)
        return False
```

### Fix C2: Update Circuit Breaker Equity Check

**Location:** Line ~223 in `_is_trading_allowed` method

Replace:
```python
account_equity = 10000  # TODO: Get from broker
```

With:
```python
account_equity = self._get_account_equity()
```

### Fix C4: Fix avoid_first_minutes Calculation

**Location:** Lines 256-264 in `_is_trading_allowed` method

Replace:
```python
avoid_first = self.config.get("avoid_first_minutes", 30)
if avoid_first > 0:
    from datetime import time as dtime
    market_open = dtime(9, 30)
    avoid_until = dtime(9, 30 + avoid_first // 60, avoid_first % 60)
    
    if market_open <= current_time_only < avoid_until:
        return (False, f"Avoiding first {avoid_first} minutes")
```

With:
```python
avoid_first = self.config.get("avoid_first_minutes", 30)
if avoid_first > 0:
    from datetime import time as dtime
    market_open_dt = datetime.combine(now_et.date(), dtime(9, 30))
    avoid_until_dt = market_open_dt + timedelta(minutes=avoid_first)
    avoid_until = avoid_until_dt.time()
    
    if dtime(9, 30) <= current_time_only < avoid_until:
        return (False, f"Avoiding first {avoid_first} minutes")
```

### Fix C5/C6: Fix Position Cleanup

**Location:** In `_check_and_manage_positions` method (around line 605-608)

Replace:
```python
# Remove closed positions
for symbol in symbols_to_close:
    if symbol in self.positions:
        del self.positions[symbol]
```

With:
```python
# Remove closed positions (only if broker confirmed close)
for symbol in symbols_to_close:
    if symbol in self.positions:
        success = self._close_position(symbol, ...)
        if success:
            del self.positions[symbol]
        # If failed, keep in positions for retry
```

**AND** update the loop logic to properly call `_close_position` and check return value.

### Fix H1: Fix Tiered Exit Percentages

**Location:** In `_check_and_manage_positions` method (around lines 570-591)

Replace the tiered exit logic:
```python
# Tier 1: 50% of remaining
self._partial_exit(position, current_price, 0.5, ...)
# Tier 2: 30% of remaining
self._partial_exit(position, current_price, 0.3, ...)
```

With:
```python
# Calculate exit quantities based on ORIGINAL position size
original_qty = position.quantity
tier1_qty = int(original_qty * 0.50)  # 50% of original
tier2_qty = int(original_qty * 0.30)  # 30% of original
tier3_qty = position.remaining_quantity  # Remaining (≈20%)

# Tier 1: Exit 50% of original
if not position.tier1_hit and current_price >= tier1_target:
    if self._partial_exit(position, current_price, tier1_qty, "Tier 1 target"):
        position.tier1_hit = True

# Tier 2: Exit 30% of original
if not position.tier2_hit and current_price >= tier2_target:
    if self._partial_exit(position, current_price, tier2_qty, "Tier 2 target"):
        position.tier2_hit = True

# Tier 3: Exit remaining
if not position.tier3_hit and current_price >= tier3_target:
    symbols_to_close.append(symbol)
    position.tier3_hit = True
```

**Note:** Need to add `tier1_hit`, `tier2_hit`, `tier3_hit` flags to Position class `__init__`:
```python
self.tier1_hit = False
self.tier2_hit = False
self.tier3_hit = False
```

### Fix H4: Standardize Timezone Handling

**Location:** Throughout the file

Replace all `datetime.now()` with:
```python
datetime.now(ZoneInfo("America/New_York"))
```

Specifically:
- Line ~115 in Position class
- Line ~252 in `_is_trading_allowed`
- Line ~631 in `_partial_exit`
- Any other bare `datetime.now()` calls

---

## CONFIG FILE UPDATES

**File:** `configs/box_trading.yaml`

Add to risk section:
```yaml
risk:
  min_risk_reward_ratio: 1.2  # Minimum R:R for trade entry
```

Add CRDO to correlation groups:
```yaml
correlation_groups:
  tech:
    - NVDA
    - MSFT
    - TSM
    - CRDO
  indices:
    - SPY
    - QQQ
  healthcare:
    - JNJ
  ev:
    - TSLA
```

---

## TESTING CHECKLIST

After applying fixes:
- [ ] Run `python runner/box_trading_runner.py` and verify no errors
- [ ] Confirm broker orders are submitted (check Alpaca dashboard)
- [ ] Verify position sizes match account equity
- [ ] Test circuit breakers trigger correctly
- [ ] Confirm correlation groups prevent duplicate exposures
- [ ] Check all Telegram alerts fire
- [ ] Run paper trading for 2-4 weeks

---

## PRIORITY

**CRITICAL - Apply immediately before next bot run**

These fixes prevent:
- Lost trades (no broker execution)
- Wrong position sizing (hardcoded equity)
- Logic errors (time calculations)
- State inconsistencies (position tracking)

---

**Next Steps:**
1. Apply all fixes to `runner/box_trading_runner.py`
2. Update `configs/box_trading.yaml`
3. Test thoroughly in paper trading
4. Review results before live trading
