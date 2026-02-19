from __future__ import annotations

"""
NEXUS Computer Use Module.

Enables NEXUS to control the computer like a human:
- Screenshot capture (via screencapture on macOS)
- Mouse click, move, scroll (via cliclick or pyautogui)
- Keyboard input (via cliclick or pyautogui)
- Open URLs in browser
- File operations
- App activation (via AppleScript)

Analyze screenshots with GLM-5/Claude vision to:
- Understand what's on screen
- Decide what action to take
- Verify action was completed

Install optional dependencies for full control:
  pip install pyautogui Pillow
  brew install cliclick  (macOS click tool, no pip needed)
"""

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class NexusComputer:
    """
    Computer use agent for NEXUS.
    Uses macOS system tools (screencapture, open, osascript, cliclick)
    with optional PyAutoGUI fallback.
    """

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = Path(artifacts_dir)
        self._ss_dir = self.artifacts_dir / "computer_use" / "screenshots"
        self._log = self.artifacts_dir / "computer_use" / "actions.jsonl"
        self._ss_dir.mkdir(parents=True, exist_ok=True)
        self._log.parent.mkdir(parents=True, exist_ok=True)
        self._has_pyautogui = self._check_pyautogui()
        self._has_cliclick = self._check_cliclick()

    def _check_pyautogui(self) -> bool:
        try:
            import pyautogui  # noqa: F401
            return True
        except ImportError:
            return False

    def _check_cliclick(self) -> bool:
        try:
            result = subprocess.run(["cliclick", "-V"], capture_output=True, timeout=3)
            return result.returncode == 0
        except Exception:
            return False

    def _log_action(self, action: str, data: Dict[str, Any], result: str) -> None:
        try:
            with open(self._log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": _now_iso(), "action": action, "data": data, "result": result
                }) + "\n")
        except Exception:
            pass

    # ── Screenshot ────────────────────────────────────────────────────────

    def screenshot(self, filename: Optional[str] = None) -> Optional[Path]:
        """
        Capture full screen screenshot.
        Returns path to PNG file, or None on failure.
        """
        ts = _now_iso()[:19].replace(":", "-")
        fname = filename or f"ss_{ts}.png"
        path = self._ss_dir / fname
        try:
            # macOS screencapture (built-in, no install)
            result = subprocess.run(
                ["screencapture", "-x", str(path)],
                capture_output=True, timeout=10,
            )
            if result.returncode == 0 and path.exists():
                self._log_action("screenshot", {"path": str(path)}, "ok")
                return path
        except Exception:
            pass
        # PyAutoGUI fallback
        if self._has_pyautogui:
            try:
                import pyautogui
                img = pyautogui.screenshot()
                img.save(str(path))
                self._log_action("screenshot", {"path": str(path)}, "pyautogui")
                return path
            except Exception:
                pass
        self._log_action("screenshot", {}, "failed")
        return None

    def screenshot_region(self, x: int, y: int, w: int, h: int) -> Optional[Path]:
        """Capture a region of the screen."""
        ts = _now_iso()[:19].replace(":", "-")
        path = self._ss_dir / f"region_{ts}.png"
        try:
            result = subprocess.run(
                ["screencapture", "-x", "-R", f"{x},{y},{w},{h}", str(path)],
                capture_output=True, timeout=10,
            )
            if result.returncode == 0:
                return path
        except Exception:
            pass
        return None

    # ── Mouse control ──────────────────────────────────────────────────────

    def click(self, x: int, y: int, button: str = "left") -> bool:
        """Click at (x, y) coordinates."""
        if self._has_cliclick:
            cmd = {"left": "c", "right": "rc", "double": "dc"}.get(button, "c")
            try:
                result = subprocess.run(
                    ["cliclick", f"{cmd}:{x},{y}"],
                    capture_output=True, timeout=5,
                )
                ok = result.returncode == 0
                self._log_action("click", {"x": x, "y": y, "button": button}, "ok" if ok else "fail")
                return ok
            except Exception:
                pass
        if self._has_pyautogui:
            try:
                import pyautogui
                btn = {"left": "left", "right": "right", "double": "left"}.get(button, "left")
                clicks = 2 if button == "double" else 1
                pyautogui.click(x, y, button=btn, clicks=clicks)
                self._log_action("click", {"x": x, "y": y}, "pyautogui")
                return True
            except Exception:
                pass
        self._log_action("click", {"x": x, "y": y}, "no_tool")
        return False

    def move(self, x: int, y: int) -> bool:
        """Move mouse to (x, y)."""
        if self._has_cliclick:
            try:
                subprocess.run(["cliclick", f"m:{x},{y}"], capture_output=True, timeout=5)
                return True
            except Exception:
                pass
        if self._has_pyautogui:
            try:
                import pyautogui
                pyautogui.moveTo(x, y)
                return True
            except Exception:
                pass
        return False

    def scroll(self, x: int, y: int, direction: str = "down", clicks: int = 3) -> bool:
        """Scroll at (x, y)."""
        if self._has_pyautogui:
            try:
                import pyautogui
                amount = -clicks if direction == "down" else clicks
                pyautogui.scroll(amount, x=x, y=y)
                return True
            except Exception:
                pass
        return False

    # ── Keyboard ──────────────────────────────────────────────────────────

    def type_text(self, text: str, interval: float = 0.03) -> bool:
        """Type text at current cursor position."""
        if self._has_cliclick:
            try:
                # cliclick types text
                result = subprocess.run(
                    ["cliclick", f"t:{text}"],
                    capture_output=True, timeout=10,
                )
                self._log_action("type", {"text": text[:30]}, "cliclick")
                return result.returncode == 0
            except Exception:
                pass
        if self._has_pyautogui:
            try:
                import pyautogui
                pyautogui.typewrite(text, interval=interval)
                self._log_action("type", {"text": text[:30]}, "pyautogui")
                return True
            except Exception:
                pass
        # AppleScript fallback (macOS)
        try:
            escaped = text.replace("\\", "\\\\").replace('"', '\\"')
            script = f'tell application "System Events" to keystroke "{escaped}"'
            subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)
            return True
        except Exception:
            pass
        return False

    def hotkey(self, *keys: str) -> bool:
        """Press a key combination (e.g. hotkey('cmd', 'c') for copy)."""
        if self._has_pyautogui:
            try:
                import pyautogui
                pyautogui.hotkey(*keys)
                return True
            except Exception:
                pass
        # AppleScript fallback for common keys
        try:
            key_map = {
                "cmd": "command", "ctrl": "control", "alt": "option",
                "shift": "shift", "enter": "return", "tab": "tab",
            }
            modifiers = [key_map.get(k, k) for k in keys[:-1]]
            key = keys[-1]
            mod_str = " & ".join(f'"{m} down"' for m in modifiers)
            script = f'tell application "System Events" to keystroke "{key}" using {{{mod_str}}}'
            subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
            return True
        except Exception:
            pass
        return False

    # ── App/browser control ───────────────────────────────────────────────

    def open_url(self, url: str, browser: str = "Google Chrome") -> bool:
        """Open a URL in browser."""
        try:
            if browser == "default":
                subprocess.Popen(["open", url])
            else:
                subprocess.Popen(["open", "-a", browser, url])
            self._log_action("open_url", {"url": url}, "ok")
            return True
        except Exception:
            try:
                subprocess.Popen(["open", url])
                return True
            except Exception:
                return False

    def open_app(self, app_name: str) -> bool:
        """Open/activate an application."""
        try:
            script = f'tell application "{app_name}" to activate'
            subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)
            self._log_action("open_app", {"app": app_name}, "ok")
            return True
        except Exception:
            return False

    def run_shell(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Run a shell command and return stdout/stderr."""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True,
                text=True, timeout=timeout,
            )
            out = {"stdout": result.stdout[:2000], "stderr": result.stderr[:500], "rc": result.returncode}
            self._log_action("shell", {"cmd": command[:80]}, f"rc={result.returncode}")
            return out
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "rc": -1}

    # ── AI-powered screen understanding ──────────────────────────────────

    def analyze_screen(self, task: str = "Describe what you see on screen") -> Dict[str, Any]:
        """
        Take screenshot and analyze with GLM-5/Claude vision.
        Returns: {description, suggested_action, screenshot_path}
        """
        ss_path = self.screenshot()
        result: Dict[str, Any] = {
            "ts": _now_iso(),
            "task": task,
            "screenshot_path": str(ss_path) if ss_path else None,
            "description": "",
            "suggested_action": "",
            "ok": False,
        }
        if not ss_path or not ss_path.exists():
            result["error"] = "Screenshot failed"
            return result

        # Read screenshot bytes → base64
        try:
            import base64
            img_bytes = ss_path.read_bytes()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        except Exception as e:
            result["error"] = f"Image read failed: {e}"
            return result

        # Call Claude/GLM-5 with vision
        try:
            import anthropic
            api_key = os.environ.get("ZAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
            base_url = os.environ.get("ZAI_ANTHROPIC_BASE_URL")
            model = os.environ.get("ZAI_DEFAULT_MODEL", "claude-sonnet-4-6")
            if not api_key:
                result["error"] = "No API key"
                return result
            kwargs: Dict[str, Any] = {"api_key": api_key, "max_retries": 1}
            if base_url:
                kwargs["base_url"] = base_url.rstrip("/")
            client = anthropic.Anthropic(**kwargs)
            resp = client.messages.create(
                model=model,
                max_tokens=600,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Task: {task}\n\n"
                                "You are NEXUS computer use agent. Analyze this screenshot and:\n"
                                "1. Describe what you see in 2-3 sentences\n"
                                "2. Suggest the next action to take (click/type/scroll/etc)\n\n"
                                "Respond as JSON: {\"description\": \"...\", \"suggested_action\": \"...\"}"
                            ),
                        },
                    ],
                }],
            )
            text = resp.content[0].text if resp.content else "{}"
            try:
                import re
                m = re.search(r"\{.*\}", text, re.DOTALL)
                parsed = json.loads(m.group(0)) if m else {}
                result["description"] = parsed.get("description", text[:300])
                result["suggested_action"] = parsed.get("suggested_action", "")
            except Exception:
                result["description"] = text[:400]
            result["ok"] = True
        except Exception as e:
            result["error"] = str(e)

        self._log_action("analyze_screen", {"task": task}, "ok" if result["ok"] else "fail")
        return result

    def auto_act(self, goal: str, max_steps: int = 5) -> List[Dict[str, Any]]:
        """
        Autonomous action loop: screenshot → analyze → act → repeat.
        Tries to accomplish 'goal' in max_steps steps.
        """
        history = []
        for step in range(max_steps):
            analysis = self.analyze_screen(
                task=f"Goal: {goal}. This is step {step+1}/{max_steps}. What action should I take?"
            )
            history.append({"step": step + 1, "analysis": analysis})
            action = analysis.get("suggested_action", "").lower()
            if not action or "done" in action or "complete" in action or "finished" in action:
                break
            # TODO: Parse and execute action from text (requires more NLP)
            time.sleep(1)
        return history

    # ── Status + history ──────────────────────────────────────────────────

    @property
    def capabilities(self) -> Dict[str, bool]:
        return {
            "screenshot": True,  # Always available on macOS
            "click": self._has_cliclick or self._has_pyautogui,
            "type": self._has_cliclick or self._has_pyautogui,
            "vision_analysis": True,  # If API key available
            "cliclick": self._has_cliclick,
            "pyautogui": self._has_pyautogui,
        }

    def recent_actions(self, limit: int = 10) -> List[Dict[str, Any]]:
        if not self._log.exists():
            return []
        lines = self._log.read_text("utf-8").splitlines()[-limit:]
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return list(reversed(out))
