"""
Klynx Agent - Tool Registry
,
"""

import os
import re
import json
import hashlib
import shutil
import shlex
import subprocess
import inspect
import time
from typing import Optional, Tuple, List, Dict, Any, get_origin, get_args, Union
from pathlib import Path


class ToolRegistry:
    ""","""
    
    #  working_dir
    working_dir = "."
    virtual_root: Optional[Path] = None
    allow_shell_commands: bool = True
    rollback_journal_root: Optional[Path] = None
    _runtime_thread_id: str = ""
    _runtime_checkpoint_id: str = ""
    SEARCH_BACKEND_REASONS = frozenset(
        {
            "system_rg",
            "managed_rg",
            "shell_fallback",
            "python_fallback",
        }
    )
    READ_BACKEND_REASONS = frozenset(
        {
            "primary_shell",
            "fallback_shell_unavailable",
            "fallback_shell_error",
            "fallback_shell_timeout",
            "forced_python_policy",
        }
    )
    
    @classmethod
    def set_working_dir(cls, working_dir: str):
        """"""
        cls.working_dir = os.path.abspath(working_dir)
        # Keep global mode truly unsandboxed. Only refresh virtual_root when
        # sandboxing is currently enabled.
        if cls.virtual_root is not None:
            cls.virtual_root = Path(cls.working_dir).resolve()

    @classmethod
    def configure_security(
        cls,
        virtual_root: Optional[str] = None,
        allow_shell_commands: Optional[bool] = None,
    ) -> None:
        """."""
        if virtual_root is not None:
            root = str(virtual_root).strip()
            if root:
                cls.virtual_root = Path(root).expanduser().resolve()
            else:
                cls.virtual_root = None
        if allow_shell_commands is not None:
            cls.allow_shell_commands = bool(allow_shell_commands)

    @classmethod
    def configure_rollback(
        cls,
        journal_root: Optional[str] = None,
    ) -> None:
        if journal_root is None:
            return
        text = str(journal_root or "").strip()
        if not text:
            cls.rollback_journal_root = None
            return
        cls.rollback_journal_root = Path(text).expanduser().resolve()

    @classmethod
    def set_runtime_context(
        cls,
        thread_id: str = "",
        checkpoint_id: str = "",
    ) -> None:
        cls._runtime_thread_id = str(thread_id or "").strip()
        cls._runtime_checkpoint_id = str(checkpoint_id or "").strip()

    @classmethod
    def _rollback_context(cls) -> Tuple[Path, str, str]:
        root = cls.rollback_journal_root
        if root is None:
            raise ValueError("rollback journal root is not configured")
        thread_id = str(cls._runtime_thread_id or "").strip()
        checkpoint_id = str(cls._runtime_checkpoint_id or "").strip()
        if not thread_id:
            raise ValueError("runtime thread_id is empty")
        if not checkpoint_id:
            raise ValueError("runtime checkpoint_id is empty")
        return root, thread_id, checkpoint_id

    @classmethod
    def _journal_paths(cls, thread_id: str) -> Tuple[Path, Path]:
        root = cls.rollback_journal_root
        if root is None:
            raise ValueError("rollback journal root is not configured")
        thread_text = str(thread_id or "").strip()
        if not thread_text:
            raise ValueError("thread_id is empty")
        thread_dir = root / thread_text
        return thread_dir, thread_dir / "journal.jsonl"

    @staticmethod
    def _safe_read_text(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    @classmethod
    def _capture_before_image(cls, path: Path) -> Tuple[bool, Optional[str]]:
        if path.exists() and path.is_file():
            return True, cls._safe_read_text(path)
        return False, None

    @classmethod
    def _persist_before_blob(cls, thread_dir: Path, content: str) -> str:
        digest = hashlib.sha1(str(content).encode("utf-8")).hexdigest()
        blob_dir = thread_dir / "blobs"
        blob_dir.mkdir(parents=True, exist_ok=True)
        blob_path = blob_dir / f"{digest}.txt"
        if not blob_path.exists():
            blob_path.write_text(str(content), encoding="utf-8")
        return str(blob_path)

    @classmethod
    def _record_before_image(
        cls,
        *,
        path: Path,
        existed: bool,
        content: Optional[str],
        tool_name: str,
    ) -> None:
        try:
            root, thread_id, checkpoint_id = cls._rollback_context()
        except Exception:
            return

        thread_dir, journal_path = cls._journal_paths(thread_id)
        thread_dir.mkdir(parents=True, exist_ok=True)

        blob_path = ""
        if existed and content is not None:
            blob_path = cls._persist_before_blob(thread_dir, content)

        record = {
            "op_seq": time.time_ns(),
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "tool": str(tool_name or "").strip(),
            "path": str(path),
            "existed": bool(existed),
            "blob_path": blob_path,
            "timestamp": int(time.time()),
        }
        with open(journal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def _git_snapshot_manifest_path(cls, thread_id: str) -> Path:
        thread_dir, _ = cls._journal_paths(thread_id)
        return thread_dir / "git_snapshots.json"

    @classmethod
    def _load_git_snapshot_manifest(cls, thread_id: str) -> Dict[str, Any]:
        path = cls._git_snapshot_manifest_path(thread_id)
        if not path.exists():
            return {"git_root": "", "entries": {}}
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return {"git_root": "", "entries": {}}
        if not isinstance(payload, dict):
            return {"git_root": "", "entries": {}}
        entries = payload.get("entries", {})
        if not isinstance(entries, dict):
            entries = {}
        return {
            "git_root": str(payload.get("git_root", "") or "").strip(),
            "entries": entries,
        }

    @classmethod
    def _save_git_snapshot_manifest(cls, thread_id: str, manifest: Dict[str, Any]) -> None:
        path = cls._git_snapshot_manifest_path(thread_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_payload = {
            "git_root": str(manifest.get("git_root", "") or "").strip(),
            "entries": dict(manifest.get("entries", {}) or {}),
        }
        path.write_text(json.dumps(safe_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def _resolve_git_root(cls) -> Tuple[Optional[Path], str]:
        try:
            proc = subprocess.run(
                ["git", "-C", str(cls.working_dir), "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=8,
            )
        except Exception as exc:
            return None, str(exc)
        if proc.returncode != 0:
            message = str(proc.stderr or proc.stdout or "not a git repository").strip()
            return None, message or "not a git repository"
        root_text = str(proc.stdout or "").strip()
        if not root_text:
            return None, "git root is empty"
        return Path(root_text).resolve(), ""

    @classmethod
    def _run_git_at_root(
        cls,
        git_root: Path,
        args: List[str],
        timeout: int = 20,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(git_root), *list(args or [])],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout,
        )

    @classmethod
    def _capture_git_checkpoint_snapshot(cls) -> None:
        try:
            _, thread_id, checkpoint_id = cls._rollback_context()
        except Exception:
            return
        if not thread_id or not checkpoint_id:
            return

        manifest = cls._load_git_snapshot_manifest(thread_id)
        entries = dict(manifest.get("entries", {}) or {})
        if checkpoint_id in entries:
            return

        git_root, git_error = cls._resolve_git_root()
        if git_root is None:
            return

        existing_root = str(manifest.get("git_root", "") or "").strip()
        if existing_root and Path(existing_root).resolve() != git_root:
            return

        head_proc = cls._run_git_at_root(git_root, ["rev-parse", "HEAD"], timeout=8)
        if head_proc.returncode != 0:
            return
        head_commit = str(head_proc.stdout or "").strip()
        if not head_commit:
            return

        stash_proc = cls._run_git_at_root(
            git_root,
            ["stash", "create", f"klynx-{thread_id}-{checkpoint_id}"],
            timeout=12,
        )
        stash_commit = str(stash_proc.stdout or "").strip() if stash_proc.returncode == 0 else ""

        entries[checkpoint_id] = {
            "head_commit": head_commit,
            "stash_commit": stash_commit,
            "captured_at": int(time.time()),
            "git_error": str(git_error or "").strip(),
        }
        manifest["git_root"] = str(git_root)
        manifest["entries"] = entries
        cls._save_git_snapshot_manifest(thread_id, manifest)

    @classmethod
    def _select_journal_records(
        cls,
        *,
        thread_id: str,
        checkpoint_set: set,
    ) -> Tuple[List[Dict[str, Any]], str]:
        try:
            _, journal_path = cls._journal_paths(thread_id)
        except Exception as exc:
            return [], str(exc)
        if not journal_path.exists():
            return [], "journal not found"

        selected: List[Dict[str, Any]] = []
        with open(journal_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = str(line or "").strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if not isinstance(item, dict):
                    continue
                checkpoint_id = str(item.get("checkpoint_id", "") or "").strip()
                if checkpoint_id in checkpoint_set:
                    selected.append(item)
        selected.sort(key=lambda item: int(item.get("op_seq", 0) or 0), reverse=True)
        return selected, ""

    @classmethod
    def rollback_workspace_with_git(
        cls,
        *,
        thread_id: str,
        target_checkpoint_id: str,
    ) -> Dict[str, Any]:
        normalized_thread = str(thread_id or "").strip()
        checkpoint_id = str(target_checkpoint_id or "").strip()
        if not normalized_thread:
            return {"ok": False, "thread_id": "", "errors": ["thread_id is empty"]}
        if not checkpoint_id:
            return {"ok": False, "thread_id": normalized_thread, "errors": ["target_checkpoint_id is empty"]}

        manifest = cls._load_git_snapshot_manifest(normalized_thread)
        entries = dict(manifest.get("entries", {}) or {})
        entry = dict(entries.get(checkpoint_id, {}) or {})
        if not entry:
            return {
                "ok": False,
                "thread_id": normalized_thread,
                "target_checkpoint_id": checkpoint_id,
                "errors": [f"git snapshot not found for checkpoint {checkpoint_id}"],
            }

        git_root_text = str(manifest.get("git_root", "") or "").strip()
        git_root = Path(git_root_text).resolve() if git_root_text else None
        resolved_git_root, resolved_error = cls._resolve_git_root()
        if git_root is None and resolved_git_root is not None:
            git_root = resolved_git_root
        if git_root is None:
            return {
                "ok": False,
                "thread_id": normalized_thread,
                "target_checkpoint_id": checkpoint_id,
                "errors": [resolved_error or "unable to resolve git root"],
            }
        if resolved_git_root is not None and resolved_git_root != git_root:
            return {
                "ok": False,
                "thread_id": normalized_thread,
                "target_checkpoint_id": checkpoint_id,
                "errors": [
                    f"git root mismatch: snapshot={git_root} current={resolved_git_root}",
                ],
            }

        head_commit = str(entry.get("head_commit", "") or "").strip()
        stash_commit = str(entry.get("stash_commit", "") or "").strip()
        if not head_commit:
            return {
                "ok": False,
                "thread_id": normalized_thread,
                "target_checkpoint_id": checkpoint_id,
                "errors": ["snapshot head_commit is empty"],
            }

        errors: List[str] = []
        reset_proc = cls._run_git_at_root(git_root, ["reset", "--hard", head_commit], timeout=40)
        if reset_proc.returncode != 0:
            errors.append(str(reset_proc.stderr or reset_proc.stdout or "git reset --hard failed").strip())

        applied_stash = False
        if not errors and stash_commit:
            apply_proc = cls._run_git_at_root(
                git_root,
                ["stash", "apply", "--index", stash_commit],
                timeout=40,
            )
            if apply_proc.returncode != 0:
                # Fallback without --index for repos that cannot restore index state cleanly.
                apply_proc = cls._run_git_at_root(
                    git_root,
                    ["stash", "apply", stash_commit],
                    timeout=40,
                )
            if apply_proc.returncode != 0:
                errors.append(str(apply_proc.stderr or apply_proc.stdout or "git stash apply failed").strip())
            else:
                applied_stash = True

        status_proc = cls._run_git_at_root(
            git_root,
            ["status", "--porcelain=v1"],
            timeout=10,
        )
        status_lines = []
        if status_proc.returncode == 0:
            status_lines = [line for line in str(status_proc.stdout or "").splitlines() if line.strip()]

        return {
            "ok": len(errors) == 0,
            "thread_id": normalized_thread,
            "target_checkpoint_id": checkpoint_id,
            "git_root": str(git_root),
            "head_commit": head_commit,
            "stash_commit": stash_commit,
            "stash_applied": applied_stash,
            "status_lines": status_lines[:40],
            "errors": errors,
        }

    @classmethod
    def rollback_workspace(
        cls,
        *,
        thread_id: str,
        rollback_checkpoint_ids: List[str],
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        checkpoint_set = {
            str(item or "").strip()
            for item in (rollback_checkpoint_ids or [])
            if str(item or "").strip()
        }
        if not checkpoint_set:
            return {
                "ok": True,
                "thread_id": str(thread_id or "").strip(),
                "records": 0,
                "restored_paths": [],
                "errors": [],
                "dry_run": bool(dry_run),
                "warning": "no rollback checkpoint ids",
            }

        selected, select_error = cls._select_journal_records(
            thread_id=thread_id,
            checkpoint_set=checkpoint_set,
        )
        if select_error and select_error != "journal not found":
            return {
                "ok": False,
                "thread_id": str(thread_id or "").strip(),
                "records": 0,
                "restored_paths": [],
                "errors": [select_error],
                "dry_run": bool(dry_run),
            }
        if select_error == "journal not found":
            return {
                "ok": True,
                "thread_id": str(thread_id or "").strip(),
                "records": 0,
                "restored_paths": [],
                "errors": [],
                "dry_run": bool(dry_run),
                "warning": "journal not found",
            }

        restored_paths: List[str] = []
        errors: List[str] = []

        for record in selected:
            path_text = str(record.get("path", "") or "").strip()
            if not path_text:
                continue
            path = Path(path_text)
            try:
                resolved = path.resolve()
                root = cls.virtual_root
                if root is not None and not cls._is_within_root(resolved, root):
                    errors.append(f"{path_text}: outside virtual root")
                    continue
                existed = bool(record.get("existed", False))
                blob_path_text = str(record.get("blob_path", "") or "").strip()
                if dry_run:
                    restored_paths.append(str(resolved))
                    continue
                if existed:
                    if not blob_path_text:
                        errors.append(f"{path_text}: missing blob")
                        continue
                    blob_path = Path(blob_path_text)
                    if not blob_path.exists():
                        errors.append(f"{path_text}: blob not found")
                        continue
                    content = cls._safe_read_text(blob_path)
                    resolved.parent.mkdir(parents=True, exist_ok=True)
                    resolved.write_text(content, encoding="utf-8")
                    restored_paths.append(str(resolved))
                else:
                    if resolved.exists():
                        if resolved.is_file():
                            resolved.unlink()
                            restored_paths.append(str(resolved))
                        else:
                            errors.append(f"{path_text}: target is not a file")
            except Exception as exc:
                errors.append(f"{path_text}: {exc}")

        return {
            "ok": len(errors) == 0,
            "thread_id": str(thread_id or "").strip(),
            "records": len(selected),
            "restored_paths": restored_paths,
            "errors": errors,
            "dry_run": bool(dry_run),
        }

    @classmethod
    def _normalize_tool_params(cls, tool_name: str, func: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        call_params = dict(params or {})
        if tool_name == "execute_command":
            if "command" not in call_params and call_params.get("cmd"):
                call_params["command"] = call_params.get("cmd")
            if "cwd" not in call_params and call_params.get("workdir"):
                call_params["cwd"] = call_params.get("workdir")
            if "timeout" not in call_params and call_params.get("timeout_ms") is not None:
                call_params["timeout_ms"] = call_params.get("timeout_ms")

        try:
            signature = inspect.signature(func)
            accepted = set(signature.parameters.keys())
            return {
                key: value
                for key, value in call_params.items()
                if key in accepted
            }
        except (TypeError, ValueError):
            return call_params

    @classmethod
    def _normalize_execute_command_alias(cls, command: str) -> Tuple[str, str]:
        """
        Normalize a small cross-platform alias set so common Linux-style commands
        work in Windows PowerShell sessions.
        """
        raw = str(command or "").strip()
        if not raw:
            return raw, ""

        lowered = raw.lower().strip()
        if os.name != "nt":
            return raw, ""

        if lowered == "ls":
            return "Get-ChildItem", "ls"
        if lowered in {"ls -la", "ls -al"}:
            return "Get-ChildItem -Force", "ls -la"
        if lowered == "ls -r":
            return "Get-ChildItem -Recurse", "ls -R"
        if lowered == "pwd":
            return "Get-Location", "pwd"

        cat_match = re.match(r"^\s*cat\s+(.+?)\s*$", raw, re.IGNORECASE)
        if cat_match:
            target = cat_match.group(1).strip()
            if target:
                return f"Get-Content {target}", "cat"

        return raw, ""

    @classmethod
    def _build_execute_command_hint(cls, command: str) -> str:
        raw = str(command or "").strip()
        if os.name != "nt" or not raw:
            return ""
        lowered = raw.lower()
        if lowered.startswith("ls"):
            return "hint: use Get-ChildItem -Force (or -Recurse)"
        if lowered.startswith("pwd"):
            return "hint: use Get-Location"
        if lowered.startswith("cat "):
            return "hint: use Get-Content <path>"
        return ""

    @staticmethod
    def _is_within_root(path: Path, root: Path) -> bool:
        try:
            return os.path.commonpath(
                [os.path.normcase(str(path)), os.path.normcase(str(root))]
            ) == os.path.normcase(str(root))
        except Exception:
            return False
    
    @classmethod
    def _resolve_path(cls, path: str) -> Path:
        ""","""
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = Path(cls.working_dir) / file_path
        resolved = file_path.resolve()
        root = cls.virtual_root
        if root is not None and not cls._is_within_root(resolved, root):
            raise ValueError(f"path escapes virtual root: {resolved} (root={root})")
        return resolved

    @classmethod
    def _read_file_lines_with_shell(cls, file_path: Path) -> Tuple[Optional[List[str]], str]:
        """
        Try shell-native file read first; return (lines, reason).
        reason must be one of READ_BACKEND_REASONS when lines is None.
        """
        if not cls.allow_shell_commands:
            return None, "forced_python_policy"

        try:
            if os.name == "nt":
                path_text = str(file_path).replace("'", "''")
                wrapped = cls._wrap_powershell_utf8_command(
                    f"Get-Content -LiteralPath '{path_text}' -Encoding UTF8 -Raw"
                )
                args = [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    wrapped,
                ]
            else:
                quoted_path = shlex.quote(str(file_path))
                args = ["/bin/bash", "-lc", f"cat -- {quoted_path}"]

            result = subprocess.run(
                args,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=12,
                cwd=str(cls.working_dir),
            )
        except FileNotFoundError:
            return None, "fallback_shell_unavailable"
        except subprocess.TimeoutExpired:
            return None, "fallback_shell_timeout"
        except Exception:
            return None, "fallback_shell_error"

        if result.returncode != 0:
            return None, "fallback_shell_error"

        content = str(result.stdout or "")
        lines = content.splitlines(keepends=True)
        return lines, "primary_shell"

    @classmethod
    def read_file(
        cls,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        mode: Optional[str] = None,
        reason: Optional[str] = None,
        hit_id: Optional[str] = None,
    ) -> str:
        """
        , slice/indentation .
        """
        try:
            file_path = cls._resolve_path(path)
            if not file_path.exists():
                return f"<error>: {path}</error>"

            if not file_path.is_file():
                return f"<error>: {path}</error>"

            lines, backend_reason = cls._read_file_lines_with_shell(file_path)
            backend = "shell"
            if lines is None:
                backend = "python"
                if backend_reason not in cls.READ_BACKEND_REASONS:
                    backend_reason = "forced_python_policy"
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

            total_lines = len(lines)
            use_offset_mode = offset is not None or limit is not None
            mode_norm = str(mode or "slice").strip().lower()
            if mode_norm not in {"slice", "indentation"}:
                mode_norm = "slice"
            reason_text = str(reason or "").strip()
            hit_text = str(hit_id or "").strip()

            def _indent_width(line: str) -> int:
                expanded = line.replace("\t", " " * 4)
                return len(expanded) - len(expanded.lstrip(" "))

            if total_lines <= 0:
                start = 0
                end = 0
                start_line_no = 0
                end_line_no = 0
                effective_limit = 0 if limit is None else max(0, int(limit or 0))
                anchor_line_no = 0
            elif mode_norm == "indentation":
                try:
                    anchor_line_no = int(start_line or 0)
                except Exception:
                    anchor_line_no = 0
                if anchor_line_no <= 0:
                    try:
                        anchor_line_no = int(offset or 0) + 1
                    except Exception:
                        anchor_line_no = 1
                anchor_line_no = max(1, min(total_lines, anchor_line_no))
                anchor_idx = anchor_line_no - 1

                try:
                    effective_limit = int(limit or 120)
                except Exception:
                    effective_limit = 120
                effective_limit = max(1, min(effective_limit, 800))

                anchor_raw = lines[anchor_idx].rstrip("\n").rstrip("\r")
                anchor_indent = _indent_width(anchor_raw) if anchor_raw.strip() else 0

                start = anchor_idx
                while start > 0:
                    prev_raw = lines[start - 1].rstrip("\n").rstrip("\r")
                    if not prev_raw.strip():
                        start -= 1
                        continue
                    if _indent_width(prev_raw) < anchor_indent:
                        break
                    start -= 1

                block_end = anchor_idx + 1
                while block_end < total_lines:
                    cur_raw = lines[block_end].rstrip("\n").rstrip("\r")
                    if not cur_raw.strip():
                        block_end += 1
                        continue
                    if _indent_width(cur_raw) < anchor_indent:
                        break
                    block_end += 1

                end = min(total_lines, max(block_end, start + 1), start + effective_limit)
                start_line_no = start + 1
                end_line_no = end
            elif use_offset_mode:
                try:
                    start = int(offset or 0)
                except Exception:
                    start = 0
                try:
                    effective_limit = int(limit or 200)
                except Exception:
                    effective_limit = 200
                start = max(0, start)
                effective_limit = max(1, effective_limit)
                end = min(total_lines, start + effective_limit)
                start_line_no = start + 1
                end_line_no = end
                anchor_line_no = start_line_no
            else:
                try:
                    start = (int(start_line) - 1) if start_line is not None else 0
                except Exception:
                    start = 0
                start = max(0, start)
                try:
                    if end_line is not None:
                        end = int(end_line)
                    elif limit is not None:
                        end = start + int(limit)
                    else:
                        end = total_lines
                except Exception:
                    end = total_lines
                end = max(start, min(total_lines, end))
                start_line_no = start + 1
                end_line_no = end
                effective_limit = max(0, end - start)
                anchor_line_no = start_line_no

            result_lines = []
            for i in range(start, end):
                line_num = i + 1
                line_content = lines[i].rstrip("\n").rstrip("\r")
                result_lines.append(f"{line_num:4d} | {line_content}")

            has_more = end < total_lines
            next_offset = end if has_more else -1
            next_start_line = end + 1 if has_more else -1

            def _escape_attr(text: str) -> str:
                return (
                    str(text or "")
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                    .replace("'", "&apos;")
                )

            abs_path = str(file_path)
            content_text = "\n".join(result_lines)
            content_hash = hashlib.sha1(content_text.encode("utf-8")).hexdigest()[:16]
            chunk_seed = f"{abs_path}:{start_line_no}:{end_line_no}:{content_hash}"
            chunk_id = hashlib.sha1(chunk_seed.encode("utf-8")).hexdigest()[:16]

            header = (
                f'<file_chunk path="{_escape_attr(abs_path)}" request_path="{_escape_attr(path)}" '
                f'total_lines="{total_lines}" start_line="{start_line_no}" end_line="{end_line_no}" '
                f'offset="{start}" limit="{effective_limit}" has_more="{str(has_more).lower()}" '
                f'next_offset="{next_offset}" next_start_line="{next_start_line}" '
                f'mode="{_escape_attr(mode_norm)}" anchor_line="{anchor_line_no}" '
                f'backend="{_escape_attr(backend)}" backend_reason="{_escape_attr(backend_reason)}" '
                f'reason="{_escape_attr(reason_text)}" hit_id="{_escape_attr(hit_text)}" '
                f'chunk_id="{chunk_id}" content_hash="{content_hash}">'
            )
            if content_text:
                return f"{header}\n{content_text}\n</file_chunk>"
            return f"{header}\n</file_chunk>"

        except Exception as e:
            return f"<error>: {str(e)}</error>"
    @classmethod
    def _prepare_patch_lines(cls, patch: str) -> List[str]:
        text = str(patch or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = text.split("\n")
        while lines and lines[0] == "":
            lines.pop(0)
        while lines and lines[-1] == "":
            lines.pop()

        if (
            len(lines) >= 2
            and lines[0].startswith("```")
            and lines[-1].strip() == "```"
        ):
            lines = lines[1:-1]
            while lines and lines[0] == "":
                lines.pop(0)
            while lines and lines[-1] == "":
                lines.pop()

        return lines

    @classmethod
    def _normalize_unified_diff_path(cls, raw_path: str) -> Optional[str]:
        path = str(raw_path or "").strip()
        if not path:
            return ""
        if "\t" in path:
            path = path.split("\t", 1)[0].strip()
        if len(path) >= 2 and path[0] == '"' and path[-1] == '"':
            path = path[1:-1]
        if path == "/dev/null":
            return None
        if path.startswith("a/") or path.startswith("b/"):
            path = path[2:]
        return path

    @classmethod
    def _parse_unified_diff_apply_patch(
        cls,
        lines: List[str],
    ) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        if not lines:
            return None, "patch  *** Begin Patch "
        if not any(line.startswith("--- ") for line in lines):
            return None, "patch  *** Begin Patch "

        ops: List[Dict[str, Any]] = []
        idx = 0
        end = len(lines)
        saw_file_header = False
        while idx < end:
            line = lines[idx]
            if not line.startswith("--- "):
                idx += 1
                continue

            saw_file_header = True
            old_path = cls._normalize_unified_diff_path(line[4:].strip())
            idx += 1
            if idx >= end or not lines[idx].startswith("+++ "):
                return None, "unified diff missing +++ file header"
            new_path = cls._normalize_unified_diff_path(lines[idx][4:].strip())
            idx += 1

            if old_path is None and new_path is None:
                return None, "unified diff has both old/new as /dev/null"

            hunk_lines: List[str] = []
            while idx < end:
                current = lines[idx]
                if current.startswith("--- "):
                    break
                if current.startswith("@@"):
                    hunk_lines.append(current if current == "@@" or current.startswith("@@ ") else "@@")
                    idx += 1
                    while idx < end:
                        hunk_line = lines[idx]
                        if hunk_line.startswith("--- ") or hunk_line.startswith("@@"):
                            break
                        if hunk_line.startswith("\\ No newline at end of file"):
                            idx += 1
                            continue
                        if not hunk_line:
                            break
                        prefix = hunk_line[0]
                        if prefix not in {" ", "+", "-"}:
                            break
                        hunk_lines.append(hunk_line)
                        idx += 1
                    continue
                idx += 1

            if old_path is None:
                if not new_path:
                    return None, "unified diff add file path is empty"
                add_lines: List[str] = []
                for hunk_line in hunk_lines:
                    if hunk_line == "@@" or hunk_line.startswith("@@ "):
                        continue
                    prefix = hunk_line[0]
                    if prefix == "-":
                        return None, f"unified diff add file includes '-' lines: {new_path}"
                    add_lines.append(hunk_line[1:])
                if not add_lines:
                    return None, f"unified diff add file has no content: {new_path}"
                ops.append({"type": "add", "path": new_path, "lines": add_lines})
                continue

            if new_path is None:
                if not old_path:
                    return None, "unified diff delete file path is empty"
                ops.append({"type": "delete", "path": old_path})
                continue

            if not old_path:
                return None, "unified diff update file path is empty"
            if not hunk_lines:
                return None, f"unified diff has no hunks for update: {old_path}"
            move_to = new_path if new_path and new_path != old_path else ""
            ops.append(
                {
                    "type": "update",
                    "path": old_path,
                    "move_to": move_to,
                    "changes": hunk_lines,
                }
            )

        if not ops:
            if saw_file_header:
                return None, "unified diff did not yield patch operations"
            return None, "patch  *** Begin Patch "
        return ops, ""

    @classmethod
    def _parse_apply_patch(cls, patch: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        lines = cls._prepare_patch_lines(patch)

        if not lines:
            return None, "patch  *** Begin Patch "

        if lines[0] != "*** Begin Patch":
            return cls._parse_unified_diff_apply_patch(lines)
        if lines[-1] != "*** End Patch":
            return None, "patch  *** End Patch "

        ops: List[Dict[str, Any]] = []
        idx = 1
        end = len(lines) - 1
        update_prefixes = ("*** Update File: ", "*** File: ")
        section_prefixes = ("*** Add File: ", "*** Delete File: ", *update_prefixes)
        while idx < end:
            line = lines[idx]
            if line.startswith("*** Add File: "):
                path = line[len("*** Add File: "):].strip()
                if not path:
                    return None, "Add File "
                idx += 1
                add_lines: List[str] = []
                while idx < end and not lines[idx].startswith("*** "):
                    body_line = lines[idx]
                    if not body_line.startswith("+"):
                        return None, f" {path}  + "
                    add_lines.append(body_line[1:])
                    idx += 1
                if not add_lines:
                    return None, f" {path} "
                ops.append({"type": "add", "path": path, "lines": add_lines})
                continue

            if line.startswith("*** Delete File: "):
                path = line[len("*** Delete File: "):].strip()
                if not path:
                    return None, "Delete File "
                ops.append({"type": "delete", "path": path})
                idx += 1
                continue

            if line.startswith(update_prefixes):
                matched_prefix = next(
                    prefix for prefix in update_prefixes if line.startswith(prefix)
                )
                path = line[len(matched_prefix):].strip()
                if not path:
                    return None, "Update File "
                idx += 1
                move_to = ""
                if idx < end and lines[idx].startswith("*** Move to: "):
                    move_to = lines[idx][len("*** Move to: "):].strip()
                    if not move_to:
                        return None, f" {path}  Move to "
                    idx += 1
                change_lines: List[str] = []
                while idx < end and not lines[idx].startswith(section_prefixes):
                    change_lines.append(lines[idx])
                    idx += 1
                if not change_lines:
                    return None, f" {path} "
                ops.append({"type": "update", "path": path, "move_to": move_to, "changes": change_lines})
                continue

            return None, f" patch : {line}"

        return ops, ""

    @classmethod
    def _split_patch_hunks(cls, change_lines: List[str]) -> Tuple[Optional[List[List[Tuple[str, str]]]], str]:
        hunks: List[List[Tuple[str, str]]] = []
        current: List[Tuple[str, str]] = []

        def flush_current():
            nonlocal current
            if current:
                hunks.append(current)
                current = []

        for raw_line in change_lines:
            if raw_line == "*** End of File":
                continue
            if raw_line in {"*** Begin Update", "*** End Update"}:
                continue
            if raw_line == "@@" or raw_line.startswith("@@ "):
                flush_current()
                continue
            if not raw_line:
                return None, "patch ;"
            prefix = raw_line[0]
            if prefix not in {" ", "+", "-"}:
                return None, f": {raw_line}"
            current.append((prefix, raw_line[1:]))
        flush_current()
        if not hunks:
            return None, "patch  hunk"
        return hunks, ""

    @classmethod
    def _find_hunk_start(
        cls,
        content_lines: List[str],
        start_index: int,
        hunk: List[Tuple[str, str]],
    ) -> int:
        required = [(tag, text) for tag, text in hunk if tag != "+"]
        if not required:
            return start_index

        max_start = len(content_lines) - len(required)
        for candidate in range(max(0, start_index), max_start + 1):
            cursor = candidate
            matched = True
            for tag, text in hunk:
                if tag == "+":
                    continue
                if cursor >= len(content_lines) or content_lines[cursor] != text:
                    matched = False
                    break
                cursor += 1
            if matched:
                return candidate
        return -1

    @classmethod
    def _apply_update_hunks(
        cls,
        original_text: str,
        change_lines: List[str],
    ) -> Tuple[Optional[str], str]:
        newline = "\r\n" if "\r\n" in original_text else "\n"
        had_trailing_newline = original_text.endswith("\n") or original_text.endswith("\r")
        content_lines = original_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if content_lines and content_lines[-1] == "":
            content_lines = content_lines[:-1]

        hunks, error = cls._split_patch_hunks(change_lines)
        if hunks is None:
            return None, error

        search_cursor = 0
        for hunk in hunks:
            start = cls._find_hunk_start(content_lines, search_cursor, hunk)
            if start < 0:
                snippet = ""
                for tag, text in hunk:
                    if tag in {" ", "-"} and text:
                        snippet = text[:80]
                        break
                return None, f" patch hunk(, patch ): {snippet or 'missing context'}"

            old_count = sum(1 for tag, _ in hunk if tag != "+")
            replacement: List[str] = []
            source_idx = start
            for tag, text in hunk:
                if tag == " ":
                    replacement.append(content_lines[source_idx])
                    source_idx += 1
                elif tag == "-":
                    source_idx += 1
                elif tag == "+":
                    replacement.append(text)
            content_lines = content_lines[:start] + replacement + content_lines[start + old_count:]
            search_cursor = start + len(replacement)

        updated_text = newline.join(content_lines)
        if had_trailing_newline or not content_lines:
            updated_text += newline
        return updated_text, ""

    @classmethod
    def _get_staged_patch_entry(
        cls,
        staged_files: Dict[str, Dict[str, Any]],
        path: Path,
    ) -> Dict[str, Any]:
        key = str(path)
        if key in staged_files:
            return staged_files[key]

        exists = path.exists()
        is_file = path.is_file() if exists else True
        content: Optional[str] = None
        if exists and is_file:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        entry = {
            "path": path,
            "exists": exists,
            "is_file": is_file,
            "content": content,
            "original_exists": exists,
            "original_is_file": is_file,
            "original_content": content,
        }
        staged_files[key] = entry
        return entry

    @classmethod
    def _stage_apply_patch_operations(
        cls,
        operations: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Dict[str, Any]]], int, int, int, int, str]:
        staged_files: Dict[str, Dict[str, Any]] = {}
        added = 0
        updated = 0
        deleted = 0
        moved = 0

        for op in operations:
            op_type = str(op.get("type", "") or "")
            if op_type == "add":
                raw_path = str(op.get("path", "") or "").strip()
                file_path = cls._resolve_path(raw_path)
                entry = cls._get_staged_patch_entry(staged_files, file_path)
                if entry["exists"]:
                    return None, 0, 0, 0, 0, f"目标文件已存在: {raw_path}"
                new_text = "\n".join(op.get("lines", []) or [])
                if new_text:
                    new_text += "\n"
                entry["exists"] = True
                entry["is_file"] = True
                entry["content"] = new_text
                added += 1
                continue

            if op_type == "delete":
                raw_path = str(op.get("path", "") or "").strip()
                file_path = cls._resolve_path(raw_path)
                entry = cls._get_staged_patch_entry(staged_files, file_path)
                if not entry["exists"]:
                    return None, 0, 0, 0, 0, f": {raw_path}"
                if not entry["is_file"]:
                    return None, 0, 0, 0, 0, f": {raw_path}"
                entry["exists"] = False
                entry["is_file"] = False
                entry["content"] = None
                deleted += 1
                continue

            if op_type == "update":
                raw_path = str(op.get("path", "") or "").strip()
                file_path = cls._resolve_path(raw_path)
                source_entry = cls._get_staged_patch_entry(staged_files, file_path)
                if not source_entry["exists"]:
                    return None, 0, 0, 0, 0, f": {raw_path}"
                if not source_entry["is_file"]:
                    return None, 0, 0, 0, 0, f": {raw_path}"

                updated_text, update_error = cls._apply_update_hunks(
                    original_text=str(source_entry.get("content", "") or ""),
                    change_lines=list(op.get("changes", []) or []),
                )
                if updated_text is None:
                    return None, 0, 0, 0, 0, update_error

                move_to = str(op.get("move_to", "") or "").strip()
                if move_to:
                    destination = cls._resolve_path(move_to)
                    if destination != file_path:
                        destination_entry = cls._get_staged_patch_entry(staged_files, destination)
                        if destination_entry["exists"]:
                            return None, 0, 0, 0, 0, f"Move to 目标文件已存在: {move_to}"
                        destination_entry["exists"] = True
                        destination_entry["is_file"] = True
                        destination_entry["content"] = updated_text
                        source_entry["exists"] = False
                        source_entry["is_file"] = False
                        source_entry["content"] = None
                        moved += 1
                    else:
                        source_entry["content"] = updated_text
                        source_entry["exists"] = True
                        source_entry["is_file"] = True
                else:
                    source_entry["content"] = updated_text
                    source_entry["exists"] = True
                    source_entry["is_file"] = True

                updated += 1
                continue

            return None, 0, 0, 0, 0, f" {op_type}"

        return staged_files, added, updated, deleted, moved, ""

    @classmethod
    def _record_staged_before_images(
        cls,
        staged_files: Dict[str, Dict[str, Any]],
    ) -> None:
        for entry in staged_files.values():
            path = entry.get("path")
            if not isinstance(path, Path):
                continue
            existed = bool(entry.get("original_exists", False)) and bool(entry.get("original_is_file", False))
            content = None
            if existed:
                content = str(entry.get("original_content", "") or "")
            cls._record_before_image(
                path=path,
                existed=existed,
                content=content,
                tool_name="apply_patch",
            )

    @classmethod
    def _commit_staged_apply_patch(
        cls,
        staged_files: Dict[str, Dict[str, Any]],
    ) -> Tuple[bool, str]:
        try:
            for entry in staged_files.values():
                if bool(entry.get("exists", False)) and bool(entry.get("is_file", False)):
                    path = entry["path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(str(entry.get("content", "") or ""))

            for entry in staged_files.values():
                if bool(entry.get("exists", False)):
                    continue
                path = entry["path"]
                if path.exists():
                    if not path.is_file():
                        raise IsADirectoryError(str(path))
                    path.unlink()
            return True, ""
        except Exception as exc:
            rollback_errors: List[str] = []
            for entry in staged_files.values():
                path = entry["path"]
                try:
                    if bool(entry.get("original_exists", False)) and bool(entry.get("original_is_file", False)):
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(str(entry.get("original_content", "") or ""))
                    else:
                        if path.exists():
                            if path.is_file():
                                path.unlink()
                            else:
                                raise IsADirectoryError(str(path))
                except Exception as rollback_exc:
                    rollback_errors.append(f"{path}: {rollback_exc}")

            error = f" patch : {exc}"
            if rollback_errors:
                error += f";: {'; '.join(rollback_errors[:3])}"
            return False, error

    @classmethod
    def apply_patch(cls, patch: str) -> str:
        """
         patch .

        :
        - Add File
        - Delete File
        - Update File
        - Move to
        """
        try:
            operations, error = cls._parse_apply_patch(patch)
            if operations is None:
                return f"<error>apply_patch : {error}</error>"
            staged_files, added, updated, deleted, moved, stage_error = cls._stage_apply_patch_operations(operations)
            if staged_files is None:
                return f"<error>apply_patch : {stage_error}</error>"

            cls._capture_git_checkpoint_snapshot()
            committed, commit_error = cls._commit_staged_apply_patch(staged_files)
            if not committed:
                return f"<error>apply_patch : {commit_error}</error>"
            cls._record_staged_before_images(staged_files)

            return (
                "<success>"
                f"apply_patch : updated={updated}, added={added}, deleted={deleted}, moved={moved}"
                "</success>"
            )
        except Exception as e:
            return f"<error>apply_patch : {str(e)}</error>"
    
    @classmethod
    def execute_command(
        cls,
        command: str,
        timeout: int = 30,
        cwd: Optional[str] = None,
        workdir: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        **_: Any,
    ) -> str:
        """
        
        
        Args:
            command: 
            timeout: ()
            cwd: 
            
        Returns:
            (stdout  stderr)
        """
        try:
            if not cls.allow_shell_commands:
                return "<error>execute_command disabled</error>"
            original_command = str(command or "")
            normalized_command, alias_source = cls._normalize_execute_command_alias(original_command)
            command = normalized_command

            #  - 
            dangerous_patterns = [
                r'rm\s+-rf\s+/',
                r'rm\s+-rf\s+~',
                r'>:\s*/dev/',
                r'dd\s+if=.*of=/dev/',
                r'\bformat\s+[a-z]:',
                r'\bdel\s+/[a-z\s]*\*',
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    return f"<error>,: {command}</error>"
            
            if timeout_ms is not None:
                try:
                    timeout_ms_int = int(timeout_ms)
                except Exception:
                    timeout_ms_int = 0
                if timeout_ms_int > 0:
                    timeout = max(1, (timeout_ms_int + 999) // 1000)

            run_cwd = cls._resolve_path(cwd or workdir or cls.working_dir)

            # (shell=False, shell )
            if os.name == "nt":
                wrapped_command = cls._wrap_powershell_utf8_command(command)
                process_args = [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    wrapped_command,
                ]
            else:
                process_args = ["/bin/bash", "-lc", command]

            result = subprocess.run(
                process_args,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=str(run_cwd),
            )
            
            output_parts = []
            if alias_source:
                output_parts.append(
                    f"[NORMALIZED COMMAND] alias={alias_source} -> {command}"
                )
            
            if result.stdout:
                output_parts.append(f"[STDOUT]\n{result.stdout}")
            
            if result.stderr:
                output_parts.append(f"[STDERR]\n{result.stderr}")
            
            if result.returncode != 0:
                output_parts.append(f"[EXIT CODE] {result.returncode}")
                hint = cls._build_execute_command_hint(original_command)
                if hint:
                    output_parts.append(f"[HINT] {hint}")
            
            return '\n'.join(output_parts) if output_parts else "<success>()</success>"
            
        except subprocess.TimeoutExpired:
            return f"<error>( {timeout} )</error>"
        except Exception as e:
            return f"<error>: {str(e)}</error>"

    @staticmethod
    def _wrap_powershell_utf8_command(command: str) -> str:
        """Wrap PowerShell command so subprocess output is emitted in UTF-8."""
        raw = str(command or "").strip()
        prefix = (
            "$OutputEncoding=[System.Text.UTF8Encoding]::new($false); "
            "[Console]::InputEncoding=[System.Text.UTF8Encoding]::new($false); "
            "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); "
            "$PSDefaultParameterValues['Out-File:Encoding']='utf8'; "
            "$PSDefaultParameterValues['Set-Content:Encoding']='utf8'; "
            "$PSDefaultParameterValues['Add-Content:Encoding']='utf8'; "
        )
        return f"{prefix}{raw}" if raw else prefix

    @staticmethod
    def _escape_xml_attr(text: Any) -> str:
        return (
            str(text or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    @classmethod
    def _read_search_context(cls, file_path: Path, line_num: int, context_lines: int) -> str:
        if context_lines <= 0:
            return ""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except (UnicodeDecodeError, PermissionError, OSError):
            return ""

        start_ctx = max(0, line_num - 1 - context_lines)
        end_ctx = min(len(lines), line_num + context_lines)
        context_slice = []
        for idx in range(start_ctx, end_ctx):
            ctx_text = lines[idx].rstrip("\n").rstrip("\r")
            context_slice.append(f"{idx + 1}: {ctx_text}")
        return "\n".join(context_slice)

    @classmethod
    def _resolve_rg_binary(cls) -> Tuple[Optional[str], str]:
        system_rg = shutil.which("rg")
        if system_rg:
            return system_rg, "system_rg"

        binary_name = "rg.exe" if os.name == "nt" else "rg"
        env_candidate = str(os.getenv("KLYNX_MANAGED_RG_PATH", "") or "").strip()
        candidates: List[Path] = []
        if env_candidate:
            candidates.append(Path(env_candidate).expanduser())

        module_root = Path(__file__).resolve().parents[2]
        candidates.append(module_root / "bin" / binary_name)
        candidates.append(Path(cls.working_dir) / "libs" / "klynx" / "klynx" / "bin" / binary_name)

        seen: set = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            key = str(resolved).lower()
            if key in seen:
                continue
            seen.add(key)
            if not resolved.exists() or not resolved.is_file():
                continue
            if os.name != "nt" and not os.access(str(resolved), os.X_OK):
                continue
            return str(resolved), "managed_rg"
        return None, ""

    @classmethod
    def _format_search_hits_output(
        cls,
        *,
        pattern: str,
        search_path: Path,
        file_pattern: str,
        backend: str,
        backend_reason: str,
        hits: List[Dict[str, Any]],
        files_searched: int,
    ) -> str:
        reason_text = (
            backend_reason
            if backend_reason in cls.SEARCH_BACKEND_REASONS
            else "python_fallback"
        )
        header = (
            f'<search_hits pattern="{cls._escape_xml_attr(pattern)}" '
            f'path="{cls._escape_xml_attr(str(search_path))}" '
            f'file_pattern="{cls._escape_xml_attr(file_pattern)}" '
            f'backend="{cls._escape_xml_attr(backend)}" '
            f'backend_reason="{cls._escape_xml_attr(reason_text)}" '
            f'files_searched="{files_searched}" hits="{len(hits)}">'
        )
        body_lines = []
        for idx, hit in enumerate(hits, 1):
            context_attr = ""
            if hit.get("context"):
                context_attr = f' context="{cls._escape_xml_attr(str(hit.get("context", "")))}"'
            body_lines.append(
                f'  <hit id="{cls._escape_xml_attr(str(hit.get("id", "")))}" rank="{idx}" '
                f'file_path="{cls._escape_xml_attr(str(hit.get("file_path", "")))}" '
                f'rel_path="{cls._escape_xml_attr(str(hit.get("rel_path", "")))}" '
                f'line="{int(hit.get("line", 0) or 0)}" score="{int(hit.get("score", 0) or 0)}"{context_attr}>'
                f'{cls._escape_xml_attr(str(hit.get("preview", "")))}'
                f"</hit>"
            )
        xml_block = header + ("\n" + "\n".join(body_lines) if body_lines else "") + "\n</search_hits>"

        if not hits:
            return xml_block + "\n<info></info>"

        summary_lines = [f" {len(hits)} ( {files_searched} )"]
        for hit in hits[: min(len(hits), 20)]:
            summary_lines.append(
                f"[{hit['id']}] {hit['rel_path']}:{hit['line']} (score={hit['score']}): {hit['preview']}"
            )
        return xml_block + "\n" + "\n".join(summary_lines)

    @classmethod
    def _search_in_files_with_rg(
        cls,
        *,
        pattern: str,
        search_path: Path,
        file_pattern: str,
        case_insensitive: bool,
        is_regex: bool,
        max_results: int,
        context_lines: int,
    ) -> Optional[str]:
        rg_bin, backend_reason = cls._resolve_rg_binary()
        if not rg_bin:
            return None

        args = [rg_bin, "--json", "--line-number", "--color", "never", "--no-heading"]
        if case_insensitive:
            args.append("-i")
        if not is_regex:
            args.append("-F")
        if file_pattern and file_pattern != "*" and search_path.is_dir():
            args.extend(["--glob", file_pattern])
        args.extend([pattern, str(search_path)])

        try:
            result = subprocess.run(
                args,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(cls.working_dir),
                timeout=20,
            )
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            return "<error>: rg timeout</error>"
        except Exception:
            return None

        if result.returncode not in (0, 1):
            stderr = str(result.stderr or "").strip()
            if is_regex and stderr:
                return f"<error>: {stderr}</error>"
            return None

        base_for_rel = search_path if search_path.is_dir() else search_path.parent
        hits: List[Dict[str, Any]] = []
        matched_files = set()
        for raw_line in str(result.stdout or "").splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if str(event.get("type", "") or "") != "match":
                continue
            data = event.get("data", {}) or {}
            path_data = data.get("path", {}) or {}
            path_text = str(path_data.get("text", "") or "").strip()
            if not path_text:
                continue

            resolved_path = Path(path_text)
            if not resolved_path.is_absolute():
                resolved_path = (Path(cls.working_dir) / resolved_path).resolve()
            else:
                resolved_path = resolved_path.resolve()
            abs_path = str(resolved_path)
            try:
                rel_path = str(resolved_path.relative_to(base_for_rel.resolve()))
            except Exception:
                rel_path = str(resolved_path.name)

            try:
                line_num = int(data.get("line_number", 0) or 0)
            except Exception:
                line_num = 0
            line_content = str((data.get("lines", {}) or {}).get("text", "") or "")
            line_content = line_content.rstrip("\n").rstrip("\r")
            submatches = data.get("submatches", []) or []
            match_start = None
            if submatches:
                try:
                    match_start = int((submatches[0] or {}).get("start", 0) or 0)
                except Exception:
                    match_start = None

            preview = line_content.strip()
            if len(preview) > 220:
                preview = preview[:220] + "..."

            score = 100
            if match_start == 0:
                score += 8
            score += max(0, 5 - min(5, line_num // 200))
            if file_pattern and file_pattern != "*":
                score += 2

            hit_seed = f"{abs_path}:{line_num}:{line_content[:180]}"
            hit_id = hashlib.sha1(hit_seed.encode("utf-8")).hexdigest()[:12]
            hit = {
                "id": hit_id,
                "file_path": abs_path,
                "rel_path": rel_path,
                "line": line_num,
                "score": score,
                "preview": preview,
            }
            if context_lines > 0:
                context = cls._read_search_context(resolved_path, line_num, context_lines)
                if context:
                    hit["context"] = context

            hits.append(hit)
            matched_files.add(abs_path)
            if len(hits) >= max_results:
                break

        hits.sort(key=lambda item: (-int(item.get("score", 0)), str(item.get("file_path", "")), int(item.get("line", 0))))
        if len(hits) > max_results:
            hits = hits[:max_results]

        return cls._format_search_hits_output(
            pattern=pattern,
            search_path=search_path,
            file_pattern=file_pattern,
            backend="rg",
            backend_reason=backend_reason,
            hits=hits,
            files_searched=len(matched_files),
        )

    @classmethod
    def _search_in_files_with_shell(
        cls,
        *,
        pattern: str,
        search_path: Path,
        file_pattern: str,
        case_insensitive: bool,
        is_regex: bool,
        max_results: int,
        context_lines: int,
        backend_reason: str,
    ) -> Tuple[Optional[str], str]:
        """
        Shell-level structured search fallback.
        Returns: (output_or_none, failure_reason_if_none)
        """
        shell_failure_reason = "fallback_shell_error"
        try:
            hits: List[Dict[str, Any]] = []
            files_seen = set()
            base_for_rel = search_path if search_path.is_dir() else search_path.parent

            if os.name == "nt":
                root_text = str(search_path).replace("'", "''")
                glob_text = str(file_pattern or "*").replace("'", "''")
                pattern_text = str(pattern or "").replace("'", "''")
                simple_match = "$false" if bool(is_regex) else "$true"
                case_sensitive = "$false" if bool(case_insensitive) else "$true"
                script = (
                    "$ErrorActionPreference='Stop';"
                    f"$root='{root_text}';$glob='{glob_text}';$pat='{pattern_text}';"
                    "if (Test-Path -LiteralPath $root -PathType Leaf) {"
                    "  $targets=@(Get-Item -LiteralPath $root);"
                    "} else {"
                    "  $targets=@(Get-ChildItem -LiteralPath $root -Recurse -File -Filter $glob -ErrorAction SilentlyContinue);"
                    "}"
                    "if (-not $targets) { exit 1 }"
                    f"$matches=$targets | Select-String -Pattern $pat -SimpleMatch:{simple_match} -CaseSensitive:{case_sensitive} -ErrorAction SilentlyContinue;"
                    "foreach ($m in $matches) {"
                    "  $line=($m.Line -replace \"`r\", \"\" -replace \"`n\", \"\");"
                    "  Write-Output (\"{0}|{1}|{2}\" -f $m.Path, $m.LineNumber, $line);"
                    "}"
                )
                wrapped = cls._wrap_powershell_utf8_command(script)
                result = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-Command",
                        wrapped,
                    ],
                    shell=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=25,
                    cwd=str(cls.working_dir),
                )
                if result.returncode not in (0, 1):
                    return None, "fallback_shell_error"
                for raw_line in str(result.stdout or "").splitlines():
                    parts = raw_line.split("|", 2)
                    if len(parts) != 3:
                        continue
                    path_text, line_text, content_text = parts
                    try:
                        line_num = int(str(line_text).strip() or "0")
                    except Exception:
                        line_num = 0
                    resolved_path = Path(path_text.strip())
                    if not resolved_path.is_absolute():
                        resolved_path = (Path(cls.working_dir) / resolved_path).resolve()
                    else:
                        resolved_path = resolved_path.resolve()
                    abs_path = str(resolved_path)
                    try:
                        rel_path = str(resolved_path.relative_to(base_for_rel.resolve()))
                    except Exception:
                        rel_path = str(resolved_path.name)
                    preview = str(content_text or "").strip()
                    if len(preview) > 220:
                        preview = preview[:220] + "..."
                    hit_seed = f"{abs_path}:{line_num}:{preview[:180]}"
                    hit_id = hashlib.sha1(hit_seed.encode("utf-8")).hexdigest()[:12]
                    hit = {
                        "id": hit_id,
                        "file_path": abs_path,
                        "rel_path": rel_path,
                        "line": line_num,
                        "score": 95,
                        "preview": preview,
                    }
                    if context_lines > 0:
                        context = cls._read_search_context(resolved_path, line_num, context_lines)
                        if context:
                            hit["context"] = context
                    hits.append(hit)
                    files_seen.add(abs_path)
                    if len(hits) >= max_results:
                        break
            else:
                grep_bin = shutil.which("grep")
                if not grep_bin:
                    return None, "fallback_shell_unavailable"
                args = [grep_bin, "-n", "-H", "--binary-files=without-match"]
                if search_path.is_dir():
                    args.append("-R")
                    for skip_dir in (".git", "__pycache__", "node_modules", ".venv", "venv"):
                        args.append(f"--exclude-dir={skip_dir}")
                if case_insensitive:
                    args.append("-i")
                if not is_regex:
                    args.append("-F")
                if file_pattern and file_pattern != "*" and search_path.is_dir():
                    args.append(f"--include={file_pattern}")
                args.append(pattern)
                args.append(str(search_path))
                result = subprocess.run(
                    args,
                    shell=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=20,
                    cwd=str(cls.working_dir),
                )
                if result.returncode not in (0, 1):
                    return None, "fallback_shell_error"
                for raw_line in str(result.stdout or "").splitlines():
                    match = re.match(r"^(.*?):(\d+):(.*)$", raw_line)
                    if not match:
                        continue
                    path_text, line_text, content_text = match.groups()
                    try:
                        line_num = int(str(line_text).strip() or "0")
                    except Exception:
                        line_num = 0
                    resolved_path = Path(path_text.strip())
                    if not resolved_path.is_absolute():
                        resolved_path = (Path(cls.working_dir) / resolved_path).resolve()
                    else:
                        resolved_path = resolved_path.resolve()
                    abs_path = str(resolved_path)
                    try:
                        rel_path = str(resolved_path.relative_to(base_for_rel.resolve()))
                    except Exception:
                        rel_path = str(resolved_path.name)
                    preview = str(content_text or "").strip()
                    if len(preview) > 220:
                        preview = preview[:220] + "..."
                    hit_seed = f"{abs_path}:{line_num}:{preview[:180]}"
                    hit_id = hashlib.sha1(hit_seed.encode("utf-8")).hexdigest()[:12]
                    hit = {
                        "id": hit_id,
                        "file_path": abs_path,
                        "rel_path": rel_path,
                        "line": line_num,
                        "score": 95,
                        "preview": preview,
                    }
                    if context_lines > 0:
                        context = cls._read_search_context(resolved_path, line_num, context_lines)
                        if context:
                            hit["context"] = context
                    hits.append(hit)
                    files_seen.add(abs_path)
                    if len(hits) >= max_results:
                        break

            hits.sort(
                key=lambda item: (
                    -int(item.get("score", 0)),
                    str(item.get("file_path", "")),
                    int(item.get("line", 0)),
                )
            )
            if len(hits) > max_results:
                hits = hits[:max_results]
            return (
                cls._format_search_hits_output(
                    pattern=pattern,
                    search_path=search_path,
                    file_pattern=file_pattern,
                    backend="shell",
                    backend_reason=backend_reason,
                    hits=hits,
                    files_searched=len(files_seen),
                ),
                "",
            )
        except subprocess.TimeoutExpired:
            shell_failure_reason = "fallback_shell_timeout"
        except FileNotFoundError:
            shell_failure_reason = "fallback_shell_unavailable"
        except Exception:
            shell_failure_reason = "fallback_shell_error"
        return None, shell_failure_reason

    @classmethod
    def _search_in_files_python(
        cls,
        *,
        pattern: str,
        search_path: Path,
        file_pattern: str,
        case_insensitive: bool,
        is_regex: bool,
        max_results: int,
        context_lines: int,
        backend_reason: str = "python_fallback",
    ) -> str:
        import fnmatch

        flags = re.IGNORECASE if case_insensitive else 0
        if is_regex:
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return f"<error>: {e}</error>"
        else:
            regex = re.compile(re.escape(pattern), flags)

        if search_path.is_file():
            files_to_search = [search_path]
            base_for_rel = search_path.parent
        else:
            files_to_search = []
            base_for_rel = search_path
            skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", ".idea", ".vscode"}
            for root, dirs, files in os.walk(search_path):
                dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
                for filename in files:
                    if fnmatch.fnmatch(filename, file_pattern):
                        files_to_search.append(Path(root) / filename)

        files_searched = 0
        hits: List[Dict[str, Any]] = []
        for file_path in files_to_search:
            if len(hits) >= max_results:
                break
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                files_searched += 1
            except (UnicodeDecodeError, PermissionError, OSError):
                continue

            abs_path = str(file_path.resolve())
            try:
                rel_path = str(file_path.resolve().relative_to(base_for_rel.resolve()))
            except Exception:
                rel_path = str(file_path.name)

            for line_num, raw_line in enumerate(lines, 1):
                if len(hits) >= max_results:
                    break
                matched = regex.search(raw_line)
                if not matched:
                    continue

                line_content = raw_line.rstrip("\n").rstrip("\r")
                line_preview = line_content.strip()
                if len(line_preview) > 220:
                    line_preview = line_preview[:220] + "..."

                score = 100
                if matched.start() == 0:
                    score += 8
                score += max(0, 5 - min(5, line_num // 200))
                if file_pattern and file_pattern != "*":
                    score += 2

                hit_seed = f"{abs_path}:{line_num}:{line_content[:180]}"
                hit_id = hashlib.sha1(hit_seed.encode("utf-8")).hexdigest()[:12]
                hit = {
                    "id": hit_id,
                    "file_path": abs_path,
                    "rel_path": rel_path,
                    "line": line_num,
                    "score": score,
                    "preview": line_preview,
                }

                if context_lines > 0:
                    start_ctx = max(0, line_num - 1 - context_lines)
                    end_ctx = min(len(lines), line_num + context_lines)
                    context_slice = []
                    for idx in range(start_ctx, end_ctx):
                        ctx_text = lines[idx].rstrip("\n").rstrip("\r")
                        context_slice.append(f"{idx + 1}: {ctx_text}")
                    hit["context"] = "\n".join(context_slice)

                hits.append(hit)

        hits.sort(key=lambda item: (-int(item.get("score", 0)), str(item.get("file_path", "")), int(item.get("line", 0))))
        if len(hits) > max_results:
            hits = hits[:max_results]

        return cls._format_search_hits_output(
            pattern=pattern,
            search_path=search_path,
            file_pattern=file_pattern,
            backend="python",
            backend_reason=backend_reason,
            hits=hits,
            files_searched=files_searched,
        )
    
    @classmethod
    def _format_directory_listing_output(
        cls,
        *,
        target_path: Path,
        depth: int,
        backend: str,
        backend_reason: str,
        entries: List[Dict[str, Any]],
    ) -> str:
        safe_reason = str(backend_reason or "").strip() or "python_fallback"
        header = (
            f'<directory_listing path="{cls._escape_xml_attr(str(target_path))}" '
            f'depth="{int(depth)}" backend="{cls._escape_xml_attr(backend)}" '
            f'backend_reason="{cls._escape_xml_attr(safe_reason)}" entries="{len(entries)}">'
        )
        body_lines: List[str] = []
        preview_lines: List[str] = []
        for idx, item in enumerate(entries, 1):
            rel_path = str(item.get("rel_path", "") or "").strip() or "."
            kind = str(item.get("type", "") or "file").strip().lower()
            if kind not in {"file", "dir"}:
                kind = "file"
            depth_val = int(item.get("depth", 0) or 0)
            entry_id = str(item.get("id", "") or "").strip() or hashlib.sha1(
                f"{rel_path}:{kind}:{depth_val}".encode("utf-8")
            ).hexdigest()[:12]
            body_lines.append(
                f'  <entry id="{cls._escape_xml_attr(entry_id)}" '
                f'rel_path="{cls._escape_xml_attr(rel_path)}" '
                f'type="{cls._escape_xml_attr(kind)}" depth="{depth_val}" />'
            )
            if idx <= 200:
                preview_lines.append(f"[{entry_id}] {kind} depth={depth_val} {rel_path}")
        xml_block = header + ("\n" + "\n".join(body_lines) if body_lines else "") + "\n</directory_listing>"
        if not preview_lines:
            return xml_block + "\n<info>no entries</info>"
        return xml_block + "\n" + "\n".join(preview_lines)

    @classmethod
    def _list_directory_with_shell(
        cls,
        *,
        target_path: Path,
        depth: int,
    ) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        if not cls.allow_shell_commands:
            return None, "python_fallback"
        depth = max(0, int(depth))
        entries: List[Dict[str, Any]] = [
            {
                "id": hashlib.sha1(f"{target_path}:dir:0".encode("utf-8")).hexdigest()[:12],
                "rel_path": ".",
                "type": "dir",
                "depth": 0,
            }
        ]
        try:
            if os.name == "nt":
                root_text = str(target_path).replace("'", "''")
                script = (
                    "$ErrorActionPreference='Stop';"
                    f"$root=(Resolve-Path -LiteralPath '{root_text}').Path;"
                    f"$maxDepth={depth};"
                    "$items = Get-ChildItem -LiteralPath $root -Force -Recurse -ErrorAction SilentlyContinue | Sort-Object FullName;"
                    "foreach ($item in $items) {"
                    "  $rel = $item.FullName.Substring($root.Length).TrimStart('\\');"
                    "  if (-not $rel) { continue }"
                    "  $depthValue = ($rel -split '\\\\').Count;"
                    "  if ($depthValue -gt $maxDepth) { continue }"
                    "  $kind = if ($item.PSIsContainer) { 'dir' } else { 'file' };"
                    "  Write-Output (\"{0}|{1}|{2}\" -f $kind, $depthValue, $rel);"
                    "}"
                )
                wrapped = cls._wrap_powershell_utf8_command(script)
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", wrapped],
                    shell=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=20,
                    cwd=str(cls.working_dir),
                )
                if result.returncode not in (0, 1):
                    return None, "python_fallback"
                for raw_line in str(result.stdout or "").splitlines():
                    parts = raw_line.split("|", 2)
                    if len(parts) != 3:
                        continue
                    kind, depth_text, rel_path = parts
                    rel_norm = str(rel_path or "").strip().replace("\\", "/")
                    if not rel_norm or rel_norm.startswith("."):
                        continue
                    try:
                        depth_val = int(str(depth_text or "").strip() or "0")
                    except Exception:
                        depth_val = 0
                    entries.append(
                        {
                            "id": hashlib.sha1(f"{rel_norm}:{kind}:{depth_val}".encode("utf-8")).hexdigest()[:12],
                            "rel_path": rel_norm,
                            "type": "dir" if str(kind).strip().lower() == "dir" else "file",
                            "depth": max(0, depth_val),
                        }
                    )
                return entries, "shell_fallback"

            quoted_root = shlex.quote(str(target_path))
            result = subprocess.run(
                ["/bin/bash", "-lc", f"cd -- {quoted_root} && find . -mindepth 1 -maxdepth {depth} -print"],
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                cwd=str(cls.working_dir),
            )
            if result.returncode not in (0, 1):
                return None, "python_fallback"
            for raw_line in str(result.stdout or "").splitlines():
                rel = str(raw_line or "").strip()
                if not rel:
                    continue
                if rel.startswith("./"):
                    rel = rel[2:]
                rel = rel.strip().replace("\\", "/")
                if not rel or rel.startswith("."):
                    continue
                abs_path = (target_path / rel).resolve()
                kind = "dir" if abs_path.is_dir() else "file"
                depth_val = rel.count("/") + 1
                entries.append(
                    {
                        "id": hashlib.sha1(f"{rel}:{kind}:{depth_val}".encode("utf-8")).hexdigest()[:12],
                        "rel_path": rel,
                        "type": kind,
                        "depth": depth_val,
                    }
                )
            return entries, "shell_fallback"
        except Exception:
            return None, "python_fallback"

    @classmethod
    def _list_directory_with_python(
        cls,
        *,
        target_path: Path,
        depth: int,
    ) -> List[Dict[str, Any]]:
        depth = max(0, int(depth))
        entries: List[Dict[str, Any]] = [
            {
                "id": hashlib.sha1(f"{target_path}:dir:0".encode("utf-8")).hexdigest()[:12],
                "rel_path": ".",
                "type": "dir",
                "depth": 0,
            }
        ]
        base = target_path.resolve()
        skip_dir_names = {".git", "__pycache__", "node_modules", ".venv", "venv", ".idea", ".vscode"}
        for root, dirs, files in os.walk(base):
            rel_root = os.path.relpath(root, base)
            root_depth = 0 if rel_root in {".", ""} else rel_root.count(os.sep) + 1
            if root_depth >= depth:
                dirs[:] = []
            else:
                dirs[:] = [d for d in dirs if d not in skip_dir_names and not d.startswith(".")]
            if root_depth > depth:
                continue
            for dir_name in sorted(dirs):
                rel = os.path.join(rel_root, dir_name) if rel_root not in {".", ""} else dir_name
                rel_norm = str(rel).replace("\\", "/")
                depth_val = rel_norm.count("/") + 1
                entries.append(
                    {
                        "id": hashlib.sha1(f"{rel_norm}:dir:{depth_val}".encode("utf-8")).hexdigest()[:12],
                        "rel_path": rel_norm,
                        "type": "dir",
                        "depth": depth_val,
                    }
                )
            for file_name in sorted(files):
                if file_name.startswith(".") and file_name not in {".gitignore", ".env.example"}:
                    continue
                rel = os.path.join(rel_root, file_name) if rel_root not in {".", ""} else file_name
                rel_norm = str(rel).replace("\\", "/")
                depth_val = rel_norm.count("/") + 1
                if depth_val > depth:
                    continue
                entries.append(
                    {
                        "id": hashlib.sha1(f"{rel_norm}:file:{depth_val}".encode("utf-8")).hexdigest()[:12],
                        "rel_path": rel_norm,
                        "type": "file",
                        "depth": depth_val,
                    }
                )
        entries.sort(key=lambda item: (int(item.get("depth", 0) or 0), str(item.get("rel_path", ""))))
        return entries

    @classmethod
    def list_directory(cls, path: str = ".", depth: int = 2) -> str:
        """
        Structured directory listing (terminal-first, cross-platform).
        """
        try:
            target_path = cls._resolve_path(path)
            if not target_path.exists():
                return f"<error>: {path}</error>"
            if not target_path.is_dir():
                return f"<error>: {path}</error>"
            safe_depth = max(0, min(int(depth or 0), 8))

            shell_entries, shell_reason = cls._list_directory_with_shell(
                target_path=target_path,
                depth=safe_depth,
            )
            if shell_entries is not None:
                return cls._format_directory_listing_output(
                    target_path=target_path,
                    depth=safe_depth,
                    backend="shell",
                    backend_reason=shell_reason or "shell_fallback",
                    entries=shell_entries,
                )

            python_entries = cls._list_directory_with_python(
                target_path=target_path,
                depth=safe_depth,
            )
            return cls._format_directory_listing_output(
                target_path=target_path,
                depth=safe_depth,
                backend="python",
                backend_reason="python_fallback",
                entries=python_entries,
            )

        except Exception as e:
            return f"<error>: {str(e)}</error>"
    @classmethod
    def search_in_files(
        cls,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*",
        case_insensitive: bool = True,
        is_regex: bool = False,
        max_results: int = 50,
        context_lines: int = 0,
        backend: str = "auto",
    ) -> str:
        """
        , hit , read_file .
        """
        try:
            search_path = cls._resolve_path(path)
            if not search_path.exists():
                return f"<error>: {path}</error>"
            backend_mode = str(backend or "auto").strip().lower()
            if backend_mode not in {"auto", "rg", "shell", "python"}:
                backend_mode = "auto"

            if backend_mode in {"auto", "rg"}:
                rg_result = cls._search_in_files_with_rg(
                    pattern=pattern,
                    search_path=search_path,
                    file_pattern=file_pattern,
                    case_insensitive=case_insensitive,
                    is_regex=is_regex,
                    max_results=max_results,
                    context_lines=context_lines,
                )
                if rg_result is not None:
                    return rg_result
                if backend_mode == "rg":
                    return cls._search_in_files_python(
                        pattern=pattern,
                        search_path=search_path,
                        file_pattern=file_pattern,
                        case_insensitive=case_insensitive,
                        is_regex=is_regex,
                        max_results=max_results,
                        context_lines=context_lines,
                        backend_reason="python_fallback",
                    )

            if backend_mode in {"auto", "shell"}:
                shell_reason = "shell_fallback"
                shell_result, shell_failure_reason = cls._search_in_files_with_shell(
                    pattern=pattern,
                    search_path=search_path,
                    file_pattern=file_pattern,
                    case_insensitive=case_insensitive,
                    is_regex=is_regex,
                    max_results=max_results,
                    context_lines=context_lines,
                    backend_reason=shell_reason,
                )
                if shell_result is not None:
                    return shell_result
                if backend_mode == "shell":
                    return cls._search_in_files_python(
                        pattern=pattern,
                        search_path=search_path,
                        file_pattern=file_pattern,
                        case_insensitive=case_insensitive,
                        is_regex=is_regex,
                        max_results=max_results,
                        context_lines=context_lines,
                        backend_reason="python_fallback",
                    )
                return cls._search_in_files_python(
                    pattern=pattern,
                    search_path=search_path,
                    file_pattern=file_pattern,
                    case_insensitive=case_insensitive,
                    is_regex=is_regex,
                    max_results=max_results,
                    context_lines=context_lines,
                    backend_reason="python_fallback",
                )

            return cls._search_in_files_python(
                pattern=pattern,
                search_path=search_path,
                file_pattern=file_pattern,
                case_insensitive=case_insensitive,
                is_regex=is_regex,
                max_results=max_results,
                context_lines=context_lines,
                backend_reason="python_fallback",
            )

        except Exception as e:
            return f"<error>: {str(e)}</error>"

    @classmethod
    def state_update(
        cls,
        overall_goal: Optional[str] = None,
        current_task: Optional[str] = None,
        task_plan: Optional[List[Dict[str, Any]]] = None,
        current_step_id: Optional[str] = None,
        completed_steps: Optional[List[str]] = None,
        blocked_reason: Optional[str] = None,
        todos: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        _ = (
            overall_goal,
            current_task,
            task_plan,
            current_step_id,
            completed_steps,
            blocked_reason,
            todos,
        )
        return "<success>state_update accepted</success>"

    @classmethod
    def execute(cls, tool_call: dict) -> str:
        """
        
        
        Args:
            tool_call: , 'tool'  'params' 
            
        Returns:
            
        """
        tool_name = tool_call.get('tool')
        params = tool_call.get('params', {})
        
        tool_map = {
            'read_file': cls.read_file,
            'apply_patch': cls.apply_patch,
            'execute_command': cls.execute_command,
            'list_directory': cls.list_directory,
            'search_in_files': cls.search_in_files,
            'state_update': cls.state_update,
        }
        
        if tool_name not in tool_map:
            return f"<error>: {tool_name}</error>"
        
        try:
            call_params = cls._normalize_tool_params(tool_name, tool_map[tool_name], params)
            return tool_map[tool_name](**call_params)
        except Exception as e:
            return f"<error>: {str(e)}</error>"


# ======================== Native Tool Calling JSON Schemas ========================

#  OpenAI tools JSON Schema (Tool Calling, Function Calling)
#  tool_call_mode="native" , Schema  LiteLLM
TOOL_SCHEMAS = {
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read focused file slices with stable chunk metadata for follow-up edits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": ","},
                    "start_line": {"type": "integer", "description": "(1)"},
                    "end_line": {"type": "integer", "description": "()"},
                    "offset": {"type": "integer", "description": "0 ,"},
                    "limit": {"type": "integer", "description": ", 80-300"},
                    "mode": {"type": "string", "description": ":slice  indentation", "default": "slice"},
                    "reason": {"type": "string", "description": "/"},
                    "hit_id": {"type": "string", "description": " search_in_files ID"}
                },
                "required": ["path"]
            }
        }
    },
    "apply_patch": {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": " patch .,,//.patch  *** Begin Patch / *** End Patch ; *** Update File:, *** File:.: @@ hunk  + ;@@ ,. unable to locate patch hunk /  patch hunk, patch , read_file  patch .",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {"type": "string", "description": " patch , apply_patch "}
                },
                "required": ["patch"]
            }
        }
    },
    "execute_command": {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Execute short-lived shell commands for build/test/git/orchestration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": ""}
                },
                "required": ["command"]
            }
        }
    },
    "list_directory": {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": ",",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": ",", "default": "."},
                    "depth": {"type": "integer", "description": ", 2", "default": 2}
                },
                "required": []
            }
        }
    },
    "search_in_files": {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": "Structured search with grep+glob style filters. Returns hit_id for read_file follow-up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": ""},
                    "path": {"type": "string", "description": ", '.'", "default": "."},
                    "file_pattern": {"type": "string", "description": ", '*.py'", "default": "*"},
                    "case_insensitive": {"type": "boolean", "description": "", "default": True},
                    "is_regex": {"type": "boolean", "description": "", "default": False},
                    "max_results": {"type": "integer", "description": "", "default": 50},
                    "context_lines": {"type": "integer", "description": "", "default": 0},
                    "backend": {
                        "type": "string",
                        "description": "Search backend policy.",
                        "enum": ["auto", "rg", "shell", "python"],
                        "default": "auto",
                    },
                },
                "required": ["pattern"]
            }
        }
    },
    "create_terminal": {
        "type": "function",
        "function": {
            "name": "create_terminal",
            "description": "..",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": ""}
                },
                "required": ["name"]
            }
        }
    },
    "run_in_terminal": {
        "type": "function",
        "function": {
            "name": "run_in_terminal",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": ""},
                    "command": {"type": "string", "description": ""}
                },
                "required": ["name", "command"]
            }
        }
    },
    "read_terminal": {
        "type": "function",
        "function": {
            "name": "read_terminal",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": ""},
                    "lines": {"type": "integer", "description": ""}
                },
                "required": ["name", "lines"]
            }
        }
    },
    "wait_terminal_until": {
        "type": "function",
        "function": {
            "name": "wait_terminal_until",
            "description": " pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": ""},
                    "op_id": {"type": "string", "description": " op_id,"},
                    "pattern": {"type": "string", "description": ","},
                    "timeout_ms": {"type": "integer", "description": "", "default": 30000},
                    "poll_interval_ms": {"type": "integer", "description": "", "default": 300}
                },
                "required": ["name"]
            }
        }
    },
    "read_terminal_since_last": {
        "type": "function",
        "function": {
            "name": "read_terminal_since_last",
            "description": ".",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": ""},
                    "op_id": {"type": "string", "description": " op_id,"},
                    "lines": {"type": "integer", "description": "", "default": 100}
                },
                "required": ["name"]
            }
        }
    },
    "run_and_wait": {
        "type": "function",
        "function": {
            "name": "run_and_wait",
            "description": " pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": ""},
                    "command": {"type": "string", "description": ""},
                    "pattern": {"type": "string", "description": ","},
                    "timeout_ms": {"type": "integer", "description": "", "default": 30000},
                    "poll_interval_ms": {"type": "integer", "description": "", "default": 300}
                },
                "required": ["name", "command"]
            }
        }
    },
    "exec_command": {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": ". REPL,shell,; session_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": ""},
                    "workdir": {"type": "string", "description": ","},
                    "tty": {"type": "boolean", "description": " PTY, true", "default": True},
                    "yield_time_ms": {"type": "integer", "description": "", "default": 1200},
                    "max_output_tokens": {"type": "integer", "description": " token ", "default": 1200},
                    "timeout_ms": {"type": "integer", "description": ",0 ", "default": 0},
                    "shell": {"type": "boolean", "description": " shell ", "default": True},
                    "login": {"type": "boolean", "description": " Unix shell  login shell ", "default": True}
                },
                "required": ["cmd"]
            }
        }
    },
    "write_stdin": {
        "type": "function",
        "function": {
            "name": "write_stdin",
            "description": " stdin, chars='' .",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": " ID"},
                    "chars": {"type": "string", "description": ";", "default": ""},
                    "yield_time_ms": {"type": "integer", "description": "", "default": 800},
                    "max_output_tokens": {"type": "integer", "description": " token ", "default": 1200}
                },
                "required": ["session_id"]
            }
        }
    },
    "close_exec_session": {
        "type": "function",
        "function": {
            "name": "close_exec_session",
            "description": ".",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": " ID"}
                },
                "required": ["session_id"]
            }
        }
    },
    "check_syntax": {
        "type": "function",
        "function": {
            "name": "check_syntax",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": ""}
                },
                "required": ["path"]
            }
        }
    },
    "open_tui": {
        "type": "function",
        "function": {
            "name": "open_tui",
            "description": " TUI ",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "TUI "},
                    "command": {"type": "string", "description": ""},
                    "rows": {"type": "integer", "description": ""},
                    "cols": {"type": "integer", "description": ""}
                },
                "required": ["name", "command"]
            }
        }
    },
    "read_tui": {
        "type": "function",
        "function": {
            "name": "read_tui",
            "description": " TUI ",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "TUI "},
                    "skip_empty_lines": {
                        "type": "boolean",
                        "description": ", true",
                        "default": True
                    }
                },
                "required": ["name"]
            }
        }
    },
    "read_tui_diff": {
        "type": "function",
        "function": {
            "name": "read_tui_diff",
            "description": " TUI .",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "TUI "}
                },
                "required": ["name"]
            }
        }
    },
    "read_tui_region": {
        "type": "function",
        "function": {
            "name": "read_tui_region",
            "description": " TUI .",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "TUI "},
                    "start_row": {"type": "integer", "description": ""},
                    "end_row": {"type": "integer", "description": ""}
                },
                "required": ["name", "start_row", "end_row"]
            }
        }
    },
    "find_text_in_tui": {
        "type": "function",
        "function": {
            "name": "find_text_in_tui",
            "description": " TUI .",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "TUI "},
                    "text": {"type": "string", "description": ""},
                    "case_insensitive": {"type": "boolean", "description": "", "default": True}
                },
                "required": ["name", "text"]
            }
        }
    },
    "send_keys": {
        "type": "function",
        "function": {
            "name": "send_keys",
            "description": " TUI ",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "TUI "},
                    "keys": {"type": "string", "description": ""}
                },
                "required": ["name", "keys"]
            }
        }
    },
    "send_keys_and_read": {
        "type": "function",
        "function": {
            "name": "send_keys_and_read",
            "description": " TUI .",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "TUI "},
                    "keys": {"type": "string", "description": ""},
                    "delay_ms": {"type": "integer", "description": "", "default": 100}
                },
                "required": ["name", "keys"]
            }
        }
    },
    "wait_tui_until": {
        "type": "function",
        "function": {
            "name": "wait_tui_until",
            "description": " TUI .",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "TUI "},
                    "text": {"type": "string", "description": ","},
                    "hash_change": {"type": "boolean", "description": "", "default": False},
                    "timeout_ms": {"type": "integer", "description": "", "default": 5000},
                    "poll_interval_ms": {"type": "integer", "description": "", "default": 200}
                },
                "required": ["name"]
            }
        }
    },
    "close_tui": {
        "type": "function",
        "function": {
            "name": "close_tui",
            "description": " TUI ",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "TUI "}
                },
                "required": ["name"]
            }
        }
    },
    "activate_tui_mode": {
        "type": "function",
        "function": {
            "name": "activate_tui_mode",
            "description": " TUI : TUI . open_tui  TUI .",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    "launch_interactive_session": {
        "type": "function",
        "function": {
            "name": "launch_interactive_session",
            "description": "(). exec_command(tty=true); TUI .",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": ", 'python', 'vim test.py'"},
                    "workdir": {"type": "string", "description": ","},
                    "yield_time_ms": {"type": "integer", "description": "", "default": 1200},
                    "max_output_tokens": {"type": "integer", "description": " token ", "default": 1200},
                    "timeout_ms": {"type": "integer", "description": ",0 ", "default": 0},
                    "shell": {"type": "boolean", "description": " shell ", "default": True},
                    "login": {"type": "boolean", "description": " Unix shell  login shell ", "default": True}
                },
                "required": ["command"]
            }
        }
    },
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": ".(,,API,,),,. search_depth='advanced' .",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": ""},
                    "max_results": {"type": "integer", "description": "", "default": 5},
                    "search_depth": {"type": "string", "description": ": 'basic'  'advanced'", "default": "basic"}
                },
                "required": ["query"]
            }
        }
    },
    "load_skill": {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": " SKILL.md ,.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "(, skill-creator)"},
                    "full": {"type": "boolean", "description": "Whether to include full SKILL.md inline. Default false for summary-only injection.", "default": False}
                },
                "required": ["name"]
            }
        }
    },
    "browser_open": {
        "type": "function",
        "function": {
            "name": "browser_open",
            "description": " URL..",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": " URL"}
                },
                "required": ["url"]
            }
        }
    },
    "browser_view": {
        "type": "function",
        "function": {
            "name": "browser_view",
            "description": ". selector .",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS ()"}
                },
                "required": []
            }
        }
    },
    "browser_act": {
        "type": "function",
        "function": {
            "name": "browser_act",
            "description": ".action  click, type, press, hover.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": ": click, type, press, hover"},
                    "selector": {"type": "string", "description": "CSS "},
                    "value": {"type": "string", "description": "(type )"}
                },
                "required": ["action", "selector"]
            }
        }
    },
    "browser_scroll": {
        "type": "function",
        "function": {
            "name": "browser_scroll",
            "description": ".direction  down, up.amount  600 .",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "description": ": down  up"},
                    "amount": {"type": "integer", "description": "()"}
                },
                "required": []
            }
        }
    },
    "browser_screenshot": {
        "type": "function",
        "function": {
            "name": "browser_screenshot",
            "description": "..",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    "browser_console_logs": {
        "type": "function",
        "function": {
            "name": "browser_console_logs",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    "state_update": {
        "type": "function",
        "function": {
            "name": "state_update",
            "description": (
                "Unified planning/task state update tool. "
                "Call only when task state actually changes. "
                "At least one of overall_goal/current_task/task_plan/current_step_id/"
                "completed_steps/blocked_reason/todos must change. "
                "Do not repeat the same state_update payload."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "overall_goal": {"type": "string", "description": "Updated overall goal text."},
                    "current_task": {"type": "string", "description": "Current focused task."},
                    "task_plan": {
                        "type": "array",
                        "description": "Plan steps. Each step can include id/title/status.",
                        "items": {"type": "object"},
                    },
                    "current_step_id": {"type": "string", "description": "Current in-progress step id."},
                    "completed_steps": {
                        "type": "array",
                        "description": "Completed step ids.",
                        "items": {"type": "string"},
                    },
                    "blocked_reason": {"type": "string", "description": "Blocker summary if blocked."},
                    "todos": {
                        "type": "array",
                        "description": "Optional todo list projection.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "content": {"type": "string"},
                                "title": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                        },
                    },
                },
                "required": []
            }
        }
    },
    "parallel_tool_call": {
        "type": "function",
        "function": {
            "name": "parallel_tool_call",
            "description": (
                "Bundle independent tool calls into one wrapper call. "
                "Only use when calls have no prerequisite dependency, do not write the same target, "
                "and failures do not affect each other."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "calls": {
                        "type": "array",
                        "description": "List of tool calls. Each item must be {tool: string, params: object}.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool": {"type": "string", "description": "Tool name."},
                                "params": {
                                    "type": "object",
                                    "description": "Tool parameters object.",
                                },
                            },
                            "required": ["tool"],
                        },
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional reason proving independence/safety of these calls.",
                    },
                },
                "required": ["calls"]
            }
        }
    },
    "run_subtask": {
        "type": "function",
        "function": {
            "name": "run_subtask",
            "description": (
                "Run a bounded sequential action list inside one subtask. "
                "Each action must be {tool: string, params: object}. "
                "Do not include run_subtask/state_update/parallel_tool_call as nested tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Optional subtask title for history and logs.",
                    },
                    "actions": {
                        "type": "array",
                        "description": "Ordered action list. Each item must include tool and optional params.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool": {
                                    "type": "string",
                                    "description": "Tool name for this action.",
                                },
                                "params": {
                                    "type": "object",
                                    "description": "Tool params object for this action.",
                                },
                            },
                            "required": ["tool"],
                        },
                    },
                },
                "required": ["actions"]
            }
        }
    }
}


def _annotation_to_json_type(annotation: Any) -> str:
    """ Python  JSON Schema ."""
    if annotation is inspect._empty:
        return "string"

    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if args:
            return _annotation_to_json_type(args[0])
        return "string"

    if annotation in (str,):
        return "string"
    if annotation in (bool,):
        return "boolean"
    if annotation in (int,):
        return "integer"
    if annotation in (float,):
        return "number"
    if annotation in (list, tuple, set):
        return "array"
    if annotation in (dict,):
        return "object"

    if origin in (list, tuple, set):
        return "array"
    if origin in (dict,):
        return "object"

    return "string"


def _build_external_tool_schema(name: str, description: str, func: Any) -> dict:
    """ JSON Schema."""
    fallback_schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description if isinstance(description, str) else f": {name}",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": ""}
                },
                "required": ["input"]
            }
        }
    }

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return fallback_schema

    properties: Dict[str, Dict[str, Any]] = {}
    required: List[str] = []

    for param_name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        json_type = _annotation_to_json_type(param.annotation)
        param_schema: Dict[str, Any] = {
            "type": json_type,
            "description": f" {param_name}",
        }
        if param.default is not inspect._empty and isinstance(param.default, (str, int, float, bool)):
            param_schema["default"] = param.default

        properties[param_name] = param_schema
        if param.default is inspect._empty:
            required.append(param_name)

    if not properties:
        return fallback_schema

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description if isinstance(description, str) else f": {name}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }
    }


def _override_tool_schema(
    name: str,
    *,
    description: Optional[str] = None,
    property_updates: Optional[Dict[str, Dict[str, Any]]] = None,
    required: Optional[List[str]] = None,
) -> None:
    schema = TOOL_SCHEMAS.get(name)
    if not isinstance(schema, dict):
        return

    fn = schema.get("function")
    if not isinstance(fn, dict):
        return

    if isinstance(description, str) and description.strip():
        fn["description"] = description

    parameters = fn.get("parameters")
    if not isinstance(parameters, dict):
        return

    properties = parameters.setdefault("properties", {})
    if not isinstance(properties, dict):
        return

    for prop_name, prop_schema in (property_updates or {}).items():
        if not isinstance(prop_schema, dict):
            continue
        existing = properties.get(prop_name, {})
        if not isinstance(existing, dict):
            existing = {}
        merged = dict(existing)
        merged.update(prop_schema)
        properties[prop_name] = merged

    if required is not None:
        parameters["required"] = list(required)


_override_tool_schema(
    "read_file",
    description=(
        "Use for precise, bounded file reads after target narrowing (line range, offset/limit, hit_id). "
        "Avoid using it as a whole-repository discovery tool."
    ),
)
_override_tool_schema(
    "apply_patch",
    description=(
        "Apply a structured minimal patch once you have a direct edit target. Matching uses exact context anchors: "
        "after each @@ block, every non-+ line must match the current file text exactly. The @@ line numbers are "
        "separators only and are not used for positioning. If you see unable to locate patch hunk, read_file the exact slice and retry a smaller patch."
    ),
)
_override_tool_schema(
    "execute_command",
    description=(
        "Use for short-lived build/test/git/runtime orchestration and one-shot verification commands. "
        "Avoid REPL/long-running foreground sessions; use exec_command for those. "
        "On Windows PowerShell, prefer workdir/cwd instead of cd /d."
    ),
    property_updates={
        "cwd": {
            "type": "string",
            "description": "Optional working directory for the command.",
        },
        "workdir": {
            "type": "string",
            "description": "Alias of cwd. Prefer passing workdir instead of embedding cd commands.",
        },
        "timeout_ms": {
            "type": "integer",
            "description": "Optional timeout in milliseconds for the command.",
            "default": 30000,
        },
    },
    required=["command"],
)
_override_tool_schema(
    "search_in_files",
    description=(
        "Use for structured grep+glob search with hit_id handoff to read_file. "
        "Avoid ad-hoc shell parsing when this tool can express the search intent directly."
    ),
)
_override_tool_schema(
    "list_directory",
    description=(
        "Use for structured directory discovery (terminal-backed listing with depth control). "
        "Avoid using execute_command ls/dir when you need machine-readable directory structure."
    ),
)
_override_tool_schema(
    "read_terminal",
    description=(
        "Read output from a legacy terminal created by create_terminal. Do not pass exec_xxx session_id values."
    ),
)
_override_tool_schema(
    "wait_terminal_until",
    description=(
        "Wait on a legacy terminal from create_terminal until a command exits or output matches a pattern. Do not pass exec_xxx session_id values."
    ),
)
_override_tool_schema(
    "read_terminal_since_last",
    description=(
        "Read incremental output from a legacy terminal created by create_terminal, not exec_command."
    ),
)
_override_tool_schema(
    "run_and_wait",
    description=(
        "Run a command in a legacy terminal and wait for completion or pattern match. Use terminal names from create_terminal, not exec_xxx session_id values."
    ),
)
_override_tool_schema(
    "exec_command",
    description=(
        "Start an interactive terminal session for REPLs, shells, or long-running foreground programs. Returns an exec_xxx session_id; continue with write_stdin or close_exec_session."
    ),
)
_override_tool_schema(
    "write_stdin",
    description=(
        "Write to an interactive exec_command session or poll for more output with chars=''. session_id must come from exec_command or launch_interactive_session."
    ),
)
_override_tool_schema(
    "close_exec_session",
    description=(
        "Close an interactive exec_command session. session_id must come from exec_command or launch_interactive_session."
    ),
)
_override_tool_schema(
    "open_tui",
    description=(
        "Start a full-screen TUI app, automatically enable TUI guidance, and return a first-screen excerpt."
    ),
    property_updates={
        "rows": {
            "type": "integer",
            "description": "Terminal rows for the TUI session.",
            "default": 30,
        },
        "cols": {
            "type": "integer",
            "description": "Terminal columns for the TUI session.",
            "default": 100,
        },
    },
)
_override_tool_schema(
    "read_tui",
    description=(
        "Read a TUI screen snapshot. When <tui_views> exists, use it as the primary TUI evidence."
    ),
)
_override_tool_schema(
    "read_tui_diff",
    description=(
        "Read the semantic diff since the last TUI observation, with concise before/after evidence."
    ),
)
_override_tool_schema(
    "read_tui_region",
    description=(
        "Read a focused row range from the current TUI screen instead of the full viewport."
    ),
)
_override_tool_schema(
    "find_text_in_tui",
    description=(
        "Find anchor text in the current TUI screen and return row-level matches for <tui_views>."
    ),
)
_override_tool_schema(
    "send_keys",
    description=(
        "Send keys to the TUI and return a short post-action excerpt when available."
    ),
)
_override_tool_schema(
    "send_keys_and_read",
    description=(
        "Send keys to the TUI and immediately return changed rows plus an after excerpt."
    ),
)
_override_tool_schema(
    "wait_tui_until",
    description=(
        "Wait until TUI text appears or the screen hash changes, and return changed-row evidence or an excerpt on success."
    ),
)
_override_tool_schema(
    "activate_tui_mode",
    description=(
        "Explicitly enable the TUI interaction guide. open_tui also enables the guide automatically."
    ),
)
_override_tool_schema(
    "launch_interactive_session",
    description=(
        "Recommended shortcut for an interactive terminal session. It is equivalent to exec_command(tty=true)."
    ),
)


def get_json_schemas(tool_names: list, external_tools: dict = None, external_tool_funcs: dict = None) -> list:
    """
    , LiteLLM/OpenAI Tool Calling  tools 
    
    Args:
        tool_names: 
        external_tools:  {name: description} ()
        external_tool_funcs:  {name: callable} ()
        
    Returns:
        OpenAI tools 
    """
    schemas = []
    for name in tool_names:
        if name in TOOL_SCHEMAS:
            schemas.append(TOOL_SCHEMAS[name])
    
    # ( callable) schema
    if external_tools:
        for name, desc in external_tools.items():
            if name not in TOOL_SCHEMAS:
                func = None
                if external_tool_funcs:
                    func = external_tool_funcs.get(name)
                if func is not None:
                    schemas.append(_build_external_tool_schema(name, desc, func))
                else:
                    schemas.append({
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": desc if isinstance(desc, str) else f": {name}",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "input": {"type": "string", "description": ""}
                                },
                                "required": ["input"]
                            }
                        }
                    })
    
    return schemas








