#!/usr/bin/env python

"""GUI launcher for the FloBaRoID robot identification pipeline.

Provides a CustomTkinter interface for running the trajectory generation,
simulation, identification, and visualization scripts with file selection
and real-time output streaming.

Usage:
  uv run gui.py
"""

import json
import os
import platform
import re
import shutil
import signal
import subprocess
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog

import customtkinter

PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODEL_DIR = PROJECT_ROOT / "model"
DATA_DIR = PROJECT_ROOT / "data"

ANSI_PATTERN = re.compile(r"\x1b\[([0-9;]*)m")

# Standard ANSI foreground colors (codes 30-37) and bright variants (90-97)
ANSI_COLORS: dict[int, str] = {
    30: "#2e2e2e",  # black
    31: "#c75050",  # red
    32: "#50a14f",  # green
    33: "#c09030",  # yellow
    34: "#4078f2",  # blue
    35: "#a626a4",  # magenta
    36: "#0897b3",  # cyan
    37: "#d0d0d0",  # white
    90: "#6a6a6a",  # bright black
    91: "#e06c75",  # bright red
    92: "#7ec87e",  # bright green
    93: "#e5c07b",  # bright yellow
    94: "#61afef",  # bright blue
    95: "#c678dd",  # bright magenta
    96: "#56b6c2",  # bright cyan
    97: "#ffffff",  # bright white
}
STATE_FILE = PROJECT_ROOT / ".gui_state.json"


class SubprocessRunner:
    """Manages non-blocking subprocess execution with real-time output streaming."""

    def __init__(
        self,
        on_output: Callable[[str], None],
        on_complete: Callable[[int], None],
    ) -> None:
        self._on_output = on_output
        self._on_complete = on_complete
        self._process: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._cancelled = threading.Event()

    def run(self, cmd: list[str]) -> None:
        """Start a subprocess and stream its output via callbacks."""
        self._cancelled.clear()
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        self._thread = threading.Thread(target=self._read_output, daemon=True)
        self._thread.start()

    def _read_output(self) -> None:
        """Read subprocess output in chunks (runs in background thread).

        Uses raw os.read() on the pipe fd so that partial lines (e.g. input()
        prompts without a trailing newline) are delivered immediately.
        """
        assert self._process is not None
        assert self._process.stdout is not None
        fd = self._process.stdout.fileno()
        while not self._cancelled.is_set():
            try:
                data = os.read(fd, 4096)
            except OSError:
                break
            if not data:
                break
            self._on_output(data.decode("utf-8", errors="replace"))
        return_code = self._process.wait()
        self._on_complete(return_code)

    def cancel(self) -> None:
        """Gracefully stop the running subprocess.

        Sends SIGINT first so Python scripts can handle KeyboardInterrupt
        and save files. Falls back to SIGTERM, then SIGKILL.
        """
        self._cancelled.set()
        if self._process is not None:
            self._process.send_signal(signal.SIGINT)
            # let the process handle KeyboardInterrupt and clean up without a timeout;
            # the reader thread will detect completion via EOF

    def send_input(self, text: str) -> None:
        """Write text to the subprocess's stdin."""
        if self._process is not None and self._process.stdin is not None and self._process.poll() is None:
            try:
                self._process.stdin.write((text + "\n").encode("utf-8"))
                self._process.stdin.flush()
            except OSError:
                pass  # process already closed

    def is_running(self) -> bool:
        """Check if a subprocess is currently running."""
        return self._process is not None and self._process.poll() is None


class PipelineRunner:
    """Sequences multiple subprocess steps, advancing on success."""

    def __init__(
        self,
        subprocess_runner: SubprocessRunner,
        on_step_start: Callable[[str], None],
        on_step_complete: Callable[[str, int], None],
        on_pipeline_complete: Callable[[bool], None],
    ) -> None:
        self._subprocess_runner = subprocess_runner
        self._on_step_start = on_step_start
        self._on_step_complete = on_step_complete
        self._on_pipeline_complete = on_pipeline_complete
        self._steps: list[tuple[str, list[str]]] = []
        self._current_index = 0

    def start(self, steps: list[tuple[str, list[str]]]) -> None:
        """Start executing a list of (label, command) steps sequentially."""
        self._steps = steps
        self._current_index = 0
        self._advance()

    def _advance(self) -> None:
        """Launch the next step or signal completion."""
        if self._current_index >= len(self._steps):
            self._on_pipeline_complete(True)
            return
        label, cmd = self._steps[self._current_index]
        self._on_step_start(label)
        self._subprocess_runner.run(cmd)

    def on_step_done(self, return_code: int) -> None:
        """Called when the current step finishes. Advances or halts."""
        label = self._steps[self._current_index][0]
        self._on_step_complete(label, return_code)
        if return_code != 0:
            self._on_pipeline_complete(False)
            return
        self._current_index += 1
        self._advance()

    def cancel(self) -> None:
        """Cancel the current step and halt the pipeline."""
        self._subprocess_runner.cancel()


class FileSelectionRow(customtkinter.CTkFrame):
    """Reusable widget: label + combobox or entry + browse button."""

    def __init__(
        self,
        master: customtkinter.CTkFrame,
        label: str,
        glob_pattern: str | None = None,
        search_dir: Path | None = None,
        filetypes: list[tuple[str, str]] | None = None,
        required: bool = False,
        allow_multiple: bool = False,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master, fg_color="transparent")
        self._filetypes = filetypes or [("All files", "*.*")]
        self._allow_multiple = allow_multiple
        self._on_change = on_change
        self._required = required

        self.grid_columnconfigure(1, weight=1)

        # label
        label_text = f"{'* ' if required else ''}{label}"
        self._label = customtkinter.CTkLabel(self, text=label_text, width=120, anchor="w")
        self._label.grid(row=0, column=0, padx=(5, 5), sticky="w")

        # populate options from glob if provided
        options: list[str] = []
        if glob_pattern and search_dir and search_dir.exists():
            options = sorted(str(p.relative_to(PROJECT_ROOT)) for p in search_dir.glob(glob_pattern))

        if options:
            self._var = customtkinter.StringVar(value="")
            self._combo = customtkinter.CTkComboBox(
                self, values=options, variable=self._var, command=self._on_combo_change
            )
            self._combo.grid(row=0, column=1, padx=5, sticky="ew")
            self._combo.set("")
            self._entry: customtkinter.CTkEntry | None = None
        else:
            self._var = customtkinter.StringVar(value="")
            self._combo = None  # type: ignore[assignment]
            self._entry = customtkinter.CTkEntry(self, textvariable=self._var)
            self._entry.grid(row=0, column=1, padx=5, sticky="ew")
            # bind key release for change detection
            self._entry.bind("<KeyRelease>", lambda _e: self._fire_change())

        # browse button
        browse_btn = customtkinter.CTkButton(self, text="Browse", width=70, command=self._browse)
        browse_btn.grid(row=0, column=2, padx=(0, 5))

    def _on_combo_change(self, _value: str) -> None:
        self._fire_change()

    def _fire_change(self) -> None:
        if self._on_change:
            self._on_change()

    def _browse(self) -> None:
        """Open a file dialog and set the result."""
        initial_dir = str(PROJECT_ROOT)
        if self._allow_multiple:
            paths = filedialog.askopenfilenames(
                initialdir=initial_dir,
                filetypes=self._filetypes,
            )
            if paths:
                # show relative paths separated by semicolons
                rel = [
                    str(Path(p).relative_to(PROJECT_ROOT)) if Path(p).is_relative_to(PROJECT_ROOT) else p for p in paths
                ]
                self._var.set("; ".join(rel))
                self._fire_change()
        else:
            path = filedialog.askopenfilename(
                initialdir=initial_dir,
                filetypes=self._filetypes,
            )
            if path:
                try:
                    rel_path = str(Path(path).relative_to(PROJECT_ROOT))
                except ValueError:
                    rel_path = path
                self._var.set(rel_path)
                self._fire_change()

    def get_value(self) -> str:
        """Return the current single file path (stripped)."""
        return self._var.get().strip()

    def get_values(self) -> list[str]:
        """Return multiple file paths (semicolon-separated, stripped)."""
        raw = self._var.get().strip()
        if not raw:
            return []
        return [p.strip() for p in raw.split(";") if p.strip()]

    def set_value(self, path: str) -> None:
        """Set the value programmatically."""
        self._var.set(path)
        self._fire_change()


class OutputPanel(customtkinter.CTkFrame):
    """Scrollable text area for subprocess output with show/hide toggle and stdin input."""

    def __init__(self, master: customtkinter.CTkBaseClass, on_input: Callable[[str], None] | None = None) -> None:
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self._visible = True
        self._on_input = on_input

        # header row
        header = customtkinter.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))
        header.grid_columnconfigure(0, weight=1)

        customtkinter.CTkLabel(header, text="Output", font=customtkinter.CTkFont(weight="bold")).grid(
            row=0, column=0, sticky="w"
        )

        self._toggle_btn = customtkinter.CTkButton(header, text="Hide", width=60, command=self._toggle)
        self._toggle_btn.grid(row=0, column=1, padx=5)

        clear_btn = customtkinter.CTkButton(header, text="Clear", width=60, command=self.clear)
        clear_btn.grid(row=0, column=2)

        # textbox
        self._textbox = customtkinter.CTkTextbox(self, height=300, font=customtkinter.CTkFont(family="monospace"))
        self._textbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=(5, 0))
        self._textbox.configure(state="disabled")
        self.grid_rowconfigure(1, weight=1)

        # configure color tags for ANSI codes
        for code, color in ANSI_COLORS.items():
            self._textbox.tag_config(f"ansi_{code}", foreground=color)
        # CTkTextbox forbids 'font' in tag_config; use underlying tk textbox for bold
        self._textbox._textbox.tag_config("ansi_bold", font=customtkinter.CTkFont(family="monospace", weight="bold"))
        self._current_tags: list[str] = []

        # input row for sending stdin to subprocess
        input_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        input_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=(2, 5))
        input_frame.grid_columnconfigure(1, weight=1)

        customtkinter.CTkLabel(input_frame, text="Input:", width=50).grid(row=0, column=0, padx=(0, 5))
        self._input_entry = customtkinter.CTkEntry(
            input_frame, placeholder_text="Type here and press Enter to send to process..."
        )
        self._input_entry.grid(row=0, column=1, sticky="ew")
        self._input_entry.bind("<Return>", self._submit_input)

        send_btn = customtkinter.CTkButton(input_frame, text="Send", width=60, command=self._submit_input)
        send_btn.grid(row=0, column=2, padx=(5, 0))

    def _submit_input(self, _event: object = None) -> None:
        """Send the input entry text to the subprocess."""
        text = self._input_entry.get()
        self._input_entry.delete(0, "end")
        self.append_text(f"{text}\n")
        if self._on_input:
            self._on_input(text)

    def _toggle(self) -> None:
        """Show or hide the output textbox."""
        self._visible = not self._visible
        if self._visible:
            self._textbox.grid()
            self._toggle_btn.configure(text="Hide")
        else:
            self._textbox.grid_remove()
            self._toggle_btn.configure(text="Show")

    def _is_scrolled_to_bottom(self) -> bool:
        """Check if the textbox is scrolled to (or near) the bottom."""
        _, bottom = self._textbox.yview()  # type: ignore[misc]
        return bottom >= 0.99

    def append_text(self, text: str) -> None:
        """Append text with ANSI color support. Only auto-scroll if already at bottom."""
        was_at_bottom = self._is_scrolled_to_bottom()
        self._textbox.configure(state="normal")
        last_end = 0
        for match in ANSI_PATTERN.finditer(text):
            # insert text before this escape sequence with current tags
            segment = text[last_end : match.start()]
            if segment:
                self._textbox.insert("end", segment, tuple(self._current_tags) if self._current_tags else ())
            last_end = match.end()

            # update current tags based on the ANSI codes
            codes_str = match.group(1)
            codes = [int(c) for c in codes_str.split(";") if c] if codes_str else [0]
            for code in codes:
                if code == 0:
                    # full reset
                    self._current_tags.clear()
                elif code == 1:
                    if "ansi_bold" not in self._current_tags:
                        self._current_tags.append("ansi_bold")
                elif code == 22:
                    # bold off
                    self._current_tags = [t for t in self._current_tags if t != "ansi_bold"]
                elif code == 39:
                    # default foreground color (Fore.RESET)
                    self._current_tags = [
                        t for t in self._current_tags if not t.startswith("ansi_3") and not t.startswith("ansi_9")
                    ]
                elif code == 49:
                    # default background color (Style.RESET_ALL background)
                    self._current_tags = [t for t in self._current_tags if not t.startswith("ansi_4")]
                elif code in ANSI_COLORS:
                    # set foreground color, replacing any existing one
                    self._current_tags = [
                        t for t in self._current_tags if not t.startswith("ansi_3") and not t.startswith("ansi_9")
                    ]
                    self._current_tags.append(f"ansi_{code}")

        # insert remaining text after the last escape sequence
        remaining = text[last_end:]
        if remaining:
            self._textbox.insert("end", remaining, tuple(self._current_tags) if self._current_tags else ())
        if was_at_bottom:
            self._textbox.see("end")
        self._textbox.configure(state="disabled")

    def clear(self) -> None:
        """Clear all text."""
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        self._textbox.configure(state="disabled")


ICON_PATH = PROJECT_ROOT / "icon.png"


def _load_app_icon() -> tk.PhotoImage | None:
    """Load the application icon from output/icon.png.

    Regenerate with: uv run tools/generate_icon.py
    """
    if ICON_PATH.exists():
        return tk.PhotoImage(file=str(ICON_PATH))
    return None


class FloBaRoIDApp(customtkinter.CTk):
    """Main application window for the FloBaRoID GUI launcher."""

    def __init__(self) -> None:
        super().__init__()

        self.title("FloBaRoID - Robot Identification Toolbox")
        self._icon = _load_app_icon()
        if self._icon:
            self.wm_iconphoto(True, self._icon)
        self.minsize(900, 650)
        self.geometry("950x750")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._pipeline_runner: PipelineRunner | None = None
        self._runner = SubprocessRunner(
            on_output=self._on_subprocess_output,
            on_complete=self._on_subprocess_complete,
        )

        self._build_file_section()
        self._build_action_section()
        self._build_output_section()
        self._load_state()
        self._update_button_states()

    # ── file selection section ──────────────────────────────────────────────

    def _build_file_section(self) -> None:
        """Create all file selection rows."""
        section = customtkinter.CTkFrame(self)
        section.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        section.grid_columnconfigure(0, weight=1)

        urdf_types = [("URDF files", "*.urdf"), ("All files", "*.*")]
        npz_types = [("NumPy archives", "*.npz"), ("All files", "*.*")]
        yaml_types = [("YAML files", "*.yaml *.yml"), ("All files", "*.*")]

        # config row with edit button
        config_frame = customtkinter.CTkFrame(section, fg_color="transparent")
        config_frame.grid(row=0, column=0, sticky="ew", pady=2)
        config_frame.grid_columnconfigure(0, weight=1)

        self._config_row = FileSelectionRow(
            config_frame,
            label="Config file:",
            glob_pattern="*.yaml",
            search_dir=CONFIGS_DIR,
            filetypes=yaml_types,
            required=True,
            on_change=self._update_button_states,
        )
        self._config_row.grid(row=0, column=0, sticky="ew")

        self._btn_edit_config = customtkinter.CTkButton(config_frame, text="Edit", width=50, command=self._edit_config)
        self._btn_edit_config.grid(row=0, column=1, padx=(5, 0))

        self._model_row = FileSelectionRow(
            section,
            label="Robot model:",
            glob_pattern="*.urdf",
            search_dir=MODEL_DIR,
            filetypes=urdf_types,
            required=True,
            on_change=self._update_button_states,
        )
        self._model_row.grid(row=1, column=0, sticky="ew", pady=2)

        self._world_row = FileSelectionRow(
            section,
            label="World model:",
            glob_pattern="world_*.urdf",
            search_dir=MODEL_DIR,
            filetypes=urdf_types,
            on_change=self._update_button_states,
        )
        self._world_row.grid(row=2, column=0, sticky="ew", pady=2)

        self._model_real_row = FileSelectionRow(
            section,
            label="Real model:",
            filetypes=urdf_types,
            on_change=self._update_button_states,
        )
        self._model_real_row.grid(row=3, column=0, sticky="ew", pady=2)

        self._trajectory_row = FileSelectionRow(
            section,
            label="Trajectory:",
            filetypes=npz_types,
            on_change=self._update_button_states,
        )
        self._trajectory_row.grid(row=4, column=0, sticky="ew", pady=2)

        self._measurements_row = FileSelectionRow(
            section,
            label="Measurements:",
            filetypes=npz_types,
            allow_multiple=True,
            on_change=self._update_button_states,
        )
        self._measurements_row.grid(row=5, column=0, sticky="ew", pady=2)

        self._validation_row = FileSelectionRow(
            section,
            label="Validation:",
            filetypes=npz_types,
            on_change=self._update_button_states,
        )
        self._validation_row.grid(row=6, column=0, sticky="ew", pady=2)

    # ── state persistence ─────────────────────────────────────────────────

    _ROW_KEYS = ("config", "model", "world", "model_real", "trajectory", "measurements", "validation")

    def _get_rows(self) -> dict[str, FileSelectionRow]:
        """Map state keys to their FileSelectionRow widgets."""
        return {
            "config": self._config_row,
            "model": self._model_row,
            "world": self._world_row,
            "model_real": self._model_real_row,
            "trajectory": self._trajectory_row,
            "measurements": self._measurements_row,
            "validation": self._validation_row,
        }

    def _load_state(self) -> None:
        """Restore file selections from the state file."""
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return
        rows = self._get_rows()
        for key, row in rows.items():
            if value := data.get(key, ""):
                row.set_value(value)

    def _save_state(self) -> None:
        """Persist current file selections to the state file."""
        rows = self._get_rows()
        data = {key: row.get_value() for key, row in rows.items()}
        try:
            STATE_FILE.write_text(json.dumps(data, indent=2) + "\n")
        except OSError:
            pass

    # ── action buttons section ──────────────────────────────────────────────

    def _build_action_section(self) -> None:
        """Create action buttons and status label."""
        section = customtkinter.CTkFrame(self)
        section.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        # top row: individual action buttons
        btn_frame = customtkinter.CTkFrame(section, fg_color="transparent")
        btn_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self._btn_trajectory = customtkinter.CTkButton(btn_frame, text="Gen. Trajectory", command=self._run_trajectory)
        self._btn_trajectory.grid(row=0, column=0, padx=5)

        self._btn_simulate = customtkinter.CTkButton(btn_frame, text="Simulate", command=self._run_simulate)
        self._btn_simulate.grid(row=0, column=1, padx=5)

        self._btn_identify = customtkinter.CTkButton(btn_frame, text="Identify", command=self._run_identify)
        self._btn_identify.grid(row=0, column=2, padx=5)

        self._btn_visualize = customtkinter.CTkButton(btn_frame, text="Visualize", command=self._run_visualize)
        self._btn_visualize.grid(row=0, column=3, padx=5)

        self._btn_cancel = customtkinter.CTkButton(btn_frame, text="Cancel", fg_color="gray", command=self._cancel)
        self._btn_cancel.grid(row=0, column=4, padx=5)

        # bottom row: pipeline button
        section.grid_columnconfigure(0, weight=1)
        self._btn_pipeline = customtkinter.CTkButton(
            section, text="Run Pipeline (Trajectory \u2192 Simulate \u2192 Identify)", command=self._run_pipeline
        )
        self._btn_pipeline.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))

        # status label
        self._status_label = customtkinter.CTkLabel(section, text="Status: Idle", anchor="w")
        self._status_label.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 5))

    # ── output section ──────────────────────────────────────────────────────

    def _build_output_section(self) -> None:
        """Create the output panel."""
        self._output_panel = OutputPanel(self, on_input=self._on_user_input)
        self._output_panel.grid(row=2, column=0, sticky="nsew", padx=10, pady=(5, 10))

    def _on_user_input(self, text: str) -> None:
        """Forward user input to the running subprocess."""
        self._runner.send_input(text)

    # ── command building ────────────────────────────────────────────────────

    def _build_command(self, script: str, extra_args: list[str] | None = None) -> list[str]:
        """Build a uv run command for the given script."""
        cmd = ["uv", "run", script]
        cmd.extend(["--config", self._config_row.get_value()])
        cmd.extend(["--model", self._model_row.get_value()])

        if script == "trajectory.py":
            if world := self._world_row.get_value():
                cmd.extend(["--world", world])
            if model_real := self._model_real_row.get_value():
                cmd.extend(["--model_real", model_real])

        elif script == "simulator.py":
            if traj := self._trajectory_row.get_value():
                cmd.extend(["--trajectory", traj])

        elif script == "identifier.py":
            for meas in self._measurements_row.get_values():
                cmd.extend(["--measurements", meas])
            if val := self._validation_row.get_value():
                cmd.extend(["--validation", val])
            if model_real := self._model_real_row.get_value():
                cmd.extend(["--model_real", model_real])

        elif script == "visualizer.py":
            # use measurements file if it's newer than the trajectory, otherwise use trajectory
            traj = self._trajectory_row.get_value()
            measurements = self._measurements_row.get_values()
            vis_file = None
            if measurements and traj:
                meas_path = PROJECT_ROOT / measurements[0]
                traj_path = PROJECT_ROOT / traj
                if meas_path.exists() and traj_path.exists() and meas_path.stat().st_mtime > traj_path.stat().st_mtime:
                    vis_file = measurements[0]
                else:
                    vis_file = traj
            elif measurements:
                vis_file = measurements[0]
            elif traj:
                vis_file = traj
            if vis_file:
                cmd.extend(["--trajectory", vis_file])
            if world := self._world_row.get_value():
                cmd.extend(["--world", world])

        if extra_args:
            cmd.extend(extra_args)

        return cmd

    # ── button state management ─────────────────────────────────────────────

    def _update_button_states(self) -> None:
        """Enable/disable buttons based on current input state and save selections."""
        has_config = bool(self._config_row.get_value())
        has_model = bool(self._model_row.get_value())
        has_measurements = len(self._measurements_row.get_values()) > 0
        running = self._runner.is_running()

        base_ok = has_config and has_model and not running

        self._btn_trajectory.configure(state="normal" if base_ok else "disabled")
        self._btn_simulate.configure(state="normal" if base_ok else "disabled")
        self._btn_identify.configure(state="normal" if base_ok and has_measurements else "disabled")
        self._btn_visualize.configure(state="normal" if base_ok else "disabled")
        self._btn_pipeline.configure(state="normal" if base_ok else "disabled")
        self._btn_cancel.configure(state="normal" if running else "disabled")

        self._save_state()

    # ── action callbacks ────────────────────────────────────────────────────

    def _run_script(self, script: str, extra_args: list[str] | None = None) -> None:
        """Run a single script as a subprocess."""
        self._pipeline_runner = None
        self._last_script = script
        cmd = self._build_command(script, extra_args)
        self._status_label.configure(text=f"Status: Running {script}...")
        self._output_panel.append_text(f"$ {' '.join(cmd)}\n")
        self._update_button_states()
        self._runner.run(cmd)
        # update button states again now that process is running
        self.after(100, self._update_button_states)

    def _edit_config(self) -> None:
        """Open the selected config file in the system's default text editor."""
        config_path = self._config_row.get_value()
        if not config_path:
            return
        full_path = str(PROJECT_ROOT / config_path)
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", full_path])
        elif system == "Windows":
            os.startfile(full_path)  # type: ignore[attr-defined]  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", full_path])

    def _run_trajectory(self) -> None:
        self._run_script("trajectory.py")

    def _run_simulate(self) -> None:
        self._run_script("simulator.py")

    def _run_identify(self) -> None:
        self._run_script("identifier.py")

    def _run_visualize(self) -> None:
        self._run_script("visualizer.py")

    def _run_pipeline(self) -> None:
        """Run the full pipeline: trajectory -> simulate -> identify."""
        model = self._model_row.get_value()
        traj_file = model + ".trajectory.npz"
        meas_file = model + ".measurements.npz"

        steps: list[tuple[str, list[str]]] = [
            ("trajectory.py", self._build_command("trajectory.py")),
            (
                "simulator.py",
                self._build_command("simulator.py", ["--trajectory", traj_file, "--filename", meas_file]),
            ),
            (
                "identifier.py",
                [
                    "uv",
                    "run",
                    "identifier.py",
                    "--config",
                    self._config_row.get_value(),
                    "--model",
                    self._model_row.get_value(),
                    "--measurements",
                    meas_file,
                ],
            ),
        ]

        self._pipeline_runner = PipelineRunner(
            subprocess_runner=self._runner,
            on_step_start=lambda label: self.after(0, self._on_pipeline_step_start, label),
            on_step_complete=lambda label, rc: None,  # handled by on_subprocess_complete
            on_pipeline_complete=lambda ok: self.after(0, self._on_pipeline_done, ok),
        )
        self._output_panel.append_text("=== Starting pipeline ===\n")
        self._pipeline_runner.start(steps)
        self.after(100, self._update_button_states)

    def _on_pipeline_step_start(self, label: str) -> None:
        self._status_label.configure(text=f"Status: Pipeline - running {label}...")
        self._output_panel.append_text(f"\n--- {label} ---\n")
        proc = self._runner._process
        if proc is not None:
            cmd_args = proc.args
            if isinstance(cmd_args, (list, tuple)):
                self._output_panel.append_text(f"$ {' '.join(str(c) for c in cmd_args)}\n")

    def _on_pipeline_done(self, success: bool) -> None:
        if success:
            self._status_label.configure(text="Status: Pipeline complete")
            self._output_panel.append_text("\n=== Pipeline finished successfully ===\n")
        else:
            self._status_label.configure(text="Status: Pipeline failed")
            self._output_panel.append_text("\n=== Pipeline failed ===\n")
        self._pipeline_runner = None
        self._update_button_states()

    def _cancel(self) -> None:
        """Cancel the running process or pipeline."""
        if self._pipeline_runner:
            self._pipeline_runner.cancel()
            self._pipeline_runner = None
        else:
            self._runner.cancel()
        self._status_label.configure(text="Status: Cancelled")
        self._output_panel.append_text("\n[Cancelled]\n")
        self.after(200, self._update_button_states)

    # ── subprocess callbacks (called from background thread) ────────────────

    def _on_subprocess_output(self, text: str) -> None:
        self.after(0, self._output_panel.append_text, text)

    def _on_subprocess_complete(self, return_code: int) -> None:
        self.after(0, self._handle_completion, return_code)

    def _handle_completion(self, return_code: int) -> None:
        """Handle subprocess completion on the main thread."""
        if return_code == 0:
            self._output_panel.append_text(f"\n[Process exited with code {return_code}]\n")
        else:
            self._output_panel.append_text(f"\n[Process FAILED with code {return_code}]\n")

        # after successful runs, auto-fill output fields if they were empty
        last = getattr(self, "_last_script", None)
        if return_code == 0 and last == "trajectory.py" and not self._trajectory_row.get_value():
            default_traj = self._model_row.get_value() + ".trajectory.npz"
            if (PROJECT_ROOT / default_traj).exists():
                self._trajectory_row.set_value(default_traj)
                self._output_panel.append_text(f"[Trajectory field set to {default_traj}]\n")

        if return_code == 0 and last == "simulator.py" and not self._measurements_row.get_value():
            default_meas = self._model_row.get_value() + ".measurements.npz"
            if (PROJECT_ROOT / default_meas).exists():
                self._measurements_row.set_value(default_meas)
                self._output_panel.append_text(f"[Measurements field set to {default_meas}]\n")

        # if we're in a pipeline, advance to the next step
        if self._pipeline_runner:
            self._pipeline_runner.on_step_done(return_code)
        else:
            status = "Complete" if return_code == 0 else f"Failed (exit code {return_code})"
            self._status_label.configure(text=f"Status: {status}")
            self._update_button_states()


def main() -> None:
    """Launch the FloBaRoID GUI application."""
    if not shutil.which("uv"):
        print("Error: 'uv' not found in PATH. Install it from https://docs.astral.sh/uv/")
        raise SystemExit(1)

    customtkinter.set_appearance_mode("system")
    customtkinter.set_default_color_theme("blue")

    app = FloBaRoIDApp()
    app.mainloop()


if __name__ == "__main__":
    main()
