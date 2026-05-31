"""Generic adapter for user-provided ``solution.py`` files.

Expected external interface for user ``solution.py`` files:

class AlgSolution:
    def reset(self, reference_text=None, reference_image=None, info=None):
        pass

    def predicts(self, observation, picked, info=None):
        return {
            "angular": 0,
            "velocity": 100,
            "viewport": 0,
            "interaction": 0,
        }

By default, ``observation`` and ``reference_image`` are base64-encoded PNG
strings, matching the competition-style Apex/R2ZeroShot solution files.
Use ``--solution-input-format raw`` to pass benchmark numpy BGR arrays directly.

The legacy ``viewport`` field is accepted in returned dicts for compatibility,
but this benchmark adapter does not map it to ``head_action`` because viewport
means up/down camera motion in the old solutions. If a solution needs benchmark
head control, return an optional ``head`` field instead.
"""

import base64
import importlib.util
import inspect
import os
import sys
from typing import Any, Dict, Tuple

from agents.agent_base import BaseAgent
from core.metrics import EpisodeMetrics


def _load_solution_class(solution_path: str, class_name: str):
    solution_path = os.path.abspath(os.path.expanduser(solution_path))
    if not os.path.isfile(solution_path):
        raise FileNotFoundError(f"solution.py not found: {solution_path}")

    solution_dir = os.path.dirname(solution_path)
    if solution_dir and solution_dir not in sys.path:
        sys.path.insert(0, solution_dir)

    module_name = f"user_solution_{abs(hash(solution_path))}"
    spec = importlib.util.spec_from_file_location(module_name, solution_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load solution module from {solution_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise AttributeError(f"{solution_path} does not define class {class_name}")
    return getattr(module, class_name)


def _numpy_to_base64_png(image: Any) -> str:
    cv2 = __import__("cv2")
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode observation as PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


class SolutionAgent(BaseAgent):
    """Adapter from user ``AlgSolution`` output to benchmark action format."""

    def __init__(
        self,
        solution_path: str = None,
        solution_class: str = "AlgSolution",
        solution_input_format: str = "base64",
        **kwargs,
    ):
        del kwargs
        self.solution_path = os.path.abspath(os.path.expanduser(solution_path or "./solution.py"))
        self.solution_class = solution_class
        self.solution_input_format = solution_input_format
        if self.solution_input_format not in ("base64", "raw"):
            raise ValueError("solution_input_format must be 'base64' or 'raw'")

        saved_cwd = os.getcwd()
        solution_dir = os.path.dirname(self.solution_path) or "."
        os.chdir(solution_dir)
        try:
            solution_cls = _load_solution_class(self.solution_path, self.solution_class)
            self.alg = solution_cls()
        finally:
            os.chdir(saved_cwd)

        self._need_reset = True
        print(
            f"[SolutionAgent] Loaded {self.solution_class} from {self.solution_path} "
            f"(input_format={self.solution_input_format})"
        )

    def reset(self):
        self._need_reset = True

    def _prepare_image(self, image: Any):
        if image is None:
            return None
        if self.solution_input_format == "raw":
            return image
        if isinstance(image, str):
            return image
        return _numpy_to_base64_png(image)

    def _maybe_reset_solution(self, info: Dict[str, Any]) -> None:
        if not self._need_reset:
            return
        reset_fn = getattr(self.alg, "reset", None)
        if callable(reset_fn):
            reference_image = self._prepare_image(info.get("reference_image"))
            saved_cwd = os.getcwd()
            solution_dir = os.path.dirname(self.solution_path) or "."
            os.chdir(solution_dir)
            try:
                self._call_reset(reset_fn, info, reference_image)
            except TypeError:
                reset_fn()
            finally:
                os.chdir(saved_cwd)
        self._need_reset = False

    @staticmethod
    def _supports_keyword(fn, keyword: str) -> bool:
        try:
            params = inspect.signature(fn).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            param.name == keyword or param.kind == inspect.Parameter.VAR_KEYWORD
            for param in params
        )

    @staticmethod
    def _supports_third_positional(fn) -> bool:
        try:
            params = inspect.signature(fn).parameters.values()
        except (TypeError, ValueError):
            return False
        positional = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        count = 0
        for param in params:
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                return True
            if param.kind in positional:
                count += 1
        return count >= 3

    def _call_reset(self, reset_fn, info: Dict[str, Any], reference_image: Any) -> None:
        if self._supports_keyword(reset_fn, "info"):
            reset_fn(
                reference_text=info.get("reference_text"),
                reference_image=reference_image,
                info=info,
            )
            return
        if self._supports_third_positional(reset_fn):
            reset_fn(info.get("reference_text"), reference_image, info)
            return
        reset_fn(
            reference_text=info.get("reference_text"),
            reference_image=reference_image,
        )

    def _prepare_observation(self, observation: Any):
        return self._prepare_image(observation)

    def _call_predicts(self, predicts, observation: Any, picked: bool, info: Dict[str, Any]):
        if self._supports_keyword(predicts, "info"):
            return predicts(observation, picked, info=info)
        if self._supports_third_positional(predicts):
            return predicts(observation, picked, info)
        return predicts(observation, picked)

    def act(self, observation: Any, info: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        self._maybe_reset_solution(info)

        predicts = getattr(self.alg, "predicts", None)
        if not callable(predicts):
            raise AttributeError(f"{self.solution_class} must define predicts(observation, picked)")

        solution_observation = self._prepare_observation(observation)
        picked = info.get("picked", False)

        saved_cwd = os.getcwd()
        solution_dir = os.path.dirname(self.solution_path) or "."
        os.chdir(solution_dir)
        try:
            solution_action = self._call_predicts(predicts, solution_observation, picked, info)
        finally:
            os.chdir(saved_cwd)

        action = self._convert_action(solution_action)
        extra_info = {
            "source": "solution",
            "solution_path": self.solution_path,
            "raw_solution_action": solution_action,
        }
        return action, extra_info

    @staticmethod
    def _convert_action(solution_action):
        np = __import__("numpy")

        if isinstance(solution_action, tuple):
            return solution_action

        if isinstance(solution_action, list):
            return tuple(solution_action)

        if not isinstance(solution_action, dict):
            raise TypeError("solution predicts() must return a dict, list, or tuple action")

        move_action = np.array(
            [
                float(solution_action.get("angular", 0.0)),
                float(solution_action.get("velocity", 0.0)),
            ],
            dtype=np.float32,
        )

        head_action = int(solution_action.get("head", 0))
        interaction = int(solution_action.get("interaction", 0))
        anim_action = 0
        if interaction == 3:
            anim_action = 3
        elif interaction == 4:
            anim_action = 4

        return move_action, head_action, anim_action

    def on_episode_end(self, success: bool, metrics: EpisodeMetrics):
        status = "SUCCESS" if success else "FAILED"
        reason = f" ({metrics.failure_reason})" if metrics.failure_reason else ""
        print(
            f"[SolutionAgent] {status}{reason} | "
            f"steps={metrics.steps}, time={metrics.time_cost:.1f}s"
        )
