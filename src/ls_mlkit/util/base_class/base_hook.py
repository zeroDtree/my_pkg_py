from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class HookStage(Enum):
    PRE_FORWARD = "pre_forward"
    POST_FORWARD = "post_forward"
    PRE_LOSS_COMPUTE = "pre_loss_compute"
    POST_LOSS_COMPUTE = "post_loss_compute"
    PRE_BACKWARD = "pre_backward"
    POST_BACKWARD = "post_backward"


class Hook:

    def __init__(
        self, name: str, stage: HookStage, fn: Callable[..., Optional[Any]], priority: int = 0, enabled: bool = True
    ):
        self.name = name
        self.stage = stage
        self.fn = fn
        self.priority = priority
        self.enabled = enabled

    def __call__(self, **kwargs) -> Optional[Any]:
        if not self.enabled:
            return None
        return self.fn(**kwargs)

    def __repr__(self):
        return f"Hook(name={self.name}, stage={self.stage}, priority={self.priority})"


class HookManager:

    def __init__(self) -> None:
        self._hooks: Dict[HookStage, List[Hook]] = {}

    def register_hook(self, hook: Hook):
        self._hooks.setdefault(hook.stage, []).append(hook)
        self._hooks[hook.stage].sort(key=lambda h: h.priority, reverse=False)

    def unregister_hook(self, name: str, stage: Optional[HookStage] = None) -> None:
        if stage:
            if stage in self._hooks:
                self._hooks[stage] = [h for h in self._hooks[stage] if h.name != name]
        else:
            for s in self._hooks:
                self._hooks[s] = [h for h in self._hooks[s] if h.name != name]

    def enable_hook(self, name: str = None, stage: HookStage = None, enabled: bool = True) -> None:
        hook_found = False
        for stage_key, hooks in self._hooks.items():
            if stage is not None and stage_key != stage:
                continue
            for h in hooks:
                if name is None or h.name == name:
                    h.enabled = enabled
                    hook_found = True
        if not hook_found:
            raise ValueError(f"Hook with name {name} not found.")

    def disable_hook(self, name: str = None, stage: HookStage = None) -> None:
        self.enable_hook(name=name, stage=stage, enabled=False)

    def run_hooks(self, stage: HookStage, **kwargs) -> Optional[Any]:
        if stage not in self._hooks:
            return None
        result = None
        for hook in self._hooks[stage]:
            if not hook.enabled:
                continue
            output = hook(**kwargs)
            if output is not None:
                result = output
        return result

    def list_hooks(self) -> None:
        for stage, hooks in self._hooks.items():
            print(f"[{stage}]")
            for h in hooks:
                print(f"  - {h} {'(enabled)' if h.enabled else '(disabled)'}")
