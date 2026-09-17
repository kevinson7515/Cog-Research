import shlex
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from colorama import Fore

from app.cosight.tool.interpreters.base import BaseInterpreter
from app.cosight.tool.interpreters.interpreter_error import InterpreterError
from app.common.logger_util import logger


class SubprocessInterpreter(BaseInterpreter):
    _CODE_EXECUTE_CMD_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "python {file_name}",
        "bash": "bash {file_name}",
        "r": "Rscript {file_name}",
    }

    _CODE_EXTENSION_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "py",
        "bash": "sh",
        "r": "R",
    }

    _CODE_TYPE_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "python",
        "py3": "python",
        "python3": "python",
        "py": "python",
        "shell": "bash",
        "bash": "bash",
        "sh": "bash",
        "r": "r",
    }

    def __init__(
        self,
        require_confirm: bool = True,
        print_stdout: bool = False,
        print_stderr: bool = True,
        timeout: int = 30,
    ) -> None:
        self.require_confirm = require_confirm
        self.print_stdout = print_stdout
        self.print_stderr = print_stderr
        self.timeout = timeout

    # ---------------------------
    # public
    # ---------------------------
    def run_file(self, file: Path, code_type: str) -> str:
        if not file.is_file():
            raise RuntimeError(f"{file} is not a file.")

        code_type = self._check_code_type(code_type)

        temp_file: Optional[Path] = None

        try:
            if code_type == "python":
                cmd, temp_file = self._prepare_python_execution(file)
            else:
                cmd = shlex.split(
                    self._CODE_EXECUTE_CMD_MAPPING[code_type].format(
                        file_name=str(file)
                    )
                )

            stdout, stderr, return_code = self._run_subprocess(cmd)

        finally:
            if temp_file and temp_file.exists():
                temp_file.unlink()

        stderr = self._filter_stderr(stderr)

        if self.print_stdout and stdout:
            logger.info("======stdout======")
            logger.info(Fore.GREEN + stdout + Fore.RESET)
            logger.info("==================")

        if self.print_stderr and stderr:
            logger.info("======stderr======")
            logger.info(Fore.RED + stderr + Fore.RESET)
            logger.info("==================")

        return self._build_result(stdout, stderr, return_code)

    def run(self, code: str, code_type: str) -> str:
        code_type = self._check_code_type(code_type)

        if self.require_confirm:
            logger.info(
                f"The following {code_type} code will run on your computer:\n{code}"
            )
            while True:
                choice = input("Running code? [Y/n]: ").lower()
                if choice in ["y", "yes", ""]:
                    break
                elif choice in ["n", "no"]:
                    raise InterpreterError("Execution halted by user.")

        temp_file = self._create_temp_file(
            code, self._CODE_EXTENSION_MAPPING[code_type]
        )

        try:
            return self.run_file(temp_file, code_type)
        finally:
            if temp_file.exists():
                temp_file.unlink()

    # ---------------------------
    # core logic
    # ---------------------------
    def _prepare_python_execution(self, file: Path):
        import ast

        with open(file, "r", encoding="utf-8") as f:
            source = f.read()

        try:
            tree = ast.parse(source)

            # 👉 自动注入 matplotlib 字体 fallback（解决 SimHei 报错）
            repo_root = Path(__file__).resolve().parents[4]
            font_candidates = [
                os.environ.get("COSIGHT_CHINESE_FONT"),
                str(repo_root / "app" / "cosight" / "tool" / "simhei.ttf"),
                str(repo_root / "app" / "cosight" / "HanSerif.ttf"),
            ]

            font_patch = ast.parse(
                f"""
import os as _cosight_os
try:
    import matplotlib as _cosight_mpl
    from matplotlib import font_manager as _cosight_fm

    _cosight_font_candidates = {font_candidates!r}
    _cosight_font_name = None
    for _cosight_font_path in _cosight_font_candidates:
        if _cosight_font_path and _cosight_os.path.exists(_cosight_font_path):
            _cosight_fm.fontManager.addfont(_cosight_font_path)
            _cosight_font_prop = _cosight_fm.FontProperties(fname=_cosight_font_path)
            _cosight_font_name = _cosight_font_prop.get_name()
            break

    if _cosight_font_name:
        _cosight_mpl.rcParams["font.family"] = [_cosight_font_name]
        _cosight_mpl.rcParams["font.sans-serif"] = [_cosight_font_name, "DejaVu Sans"]
    else:
        _cosight_mpl.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
    _cosight_mpl.rcParams["axes.unicode_minus"] = False
except Exception:
    pass
"""
            )
            tree.body = font_patch.body + tree.body

            if tree.body:
                last = tree.body[-1]

                if isinstance(last, ast.Expr):
                    tree.body[-1] = ast.Expr(
                        value=ast.Call(
                            func=ast.Name(id="print", ctx=ast.Load()),
                            args=[
                                ast.Call(
                                    func=ast.Name(id="repr", ctx=ast.Load()),
                                    args=[last.value],
                                    keywords=[],
                                )
                            ],
                            keywords=[],
                        )
                    )

            ast.fix_missing_locations(tree)

            try:
                # Python 3.9+
                modified_source = ast.unparse(tree)
            except Exception:
                import astor

                modified_source = astor.to_source(tree)

            temp_file = self._create_temp_file(modified_source, "py")
            cmd = ["python", str(temp_file)]

            return cmd, temp_file

        except Exception:
            # fallback
            cmd = ["python", str(file)]
            return cmd, None

    def _run_subprocess(self, cmd):
        try:
            env = os.environ.copy()
            env.setdefault("MPLBACKEND", "Agg")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            stdout, stderr = proc.communicate(timeout=self.timeout)

            return stdout, stderr, proc.returncode

        except subprocess.TimeoutExpired:
            proc.kill()
            return "", "Execution timed out", -1

        except Exception as e:
            return "", str(e), -1

    # ---------------------------
    # utils
    # ---------------------------
    def _filter_stderr(self, stderr: str) -> str:
        if not stderr:
            return ""

        filtered = []
        for line in stderr.splitlines():
            if "findfont" in line:
                continue
            if "Glyph " in line and "missing from font" in line:
                continue
            filtered.append(line)

        return "\n".join(filtered)

    def _build_result(self, stdout, stderr, return_code):
        result = ""

        if stdout:
            result += stdout

        if stderr:
            result += f"(stderr: {stderr})"

        if return_code != 0:
            msg = f"(Execution failed with return code {return_code})"
            if msg not in result:
                result += msg

        return result

    def _create_temp_file(self, code: str, extension: str) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=f".{extension}", encoding="utf-8"
        ) as f:
            f.write(code)
            return Path(f.name)

    def _check_code_type(self, code_type: str) -> str:
        if code_type not in self._CODE_TYPE_MAPPING:
            raise InterpreterError(
                f"Unsupported code type {code_type}. "
                f"Supported: {', '.join(self._CODE_EXTENSION_MAPPING.keys())}"
            )
        return self._CODE_TYPE_MAPPING[code_type]

    def supported_code_types(self) -> List[str]:
        return list(self._CODE_EXTENSION_MAPPING.keys())

    def update_action_space(self, action_space: Dict[str, Any]) -> None:
        raise RuntimeError("SubprocessInterpreter doesn't support action_space.")
