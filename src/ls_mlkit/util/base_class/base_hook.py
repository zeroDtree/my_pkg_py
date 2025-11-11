from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

HookStageType = TypeVar("HookStageType", bound=Enum)


class Hook(Generic[HookStageType]):

    def __init__(
        self, name: str, stage: HookStageType, fn: Callable[..., Optional[Any]], priority: int = 0, enabled: bool = True
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


class HookManager(Generic[HookStageType]):

    def __init__(self) -> None:
        self._hooks: Dict[HookStageType, List[Hook[HookStageType]]] = {}

    def register_hook(self, hook: Hook):
        self._hooks.setdefault(hook.stage, []).append(hook)
        self._hooks[hook.stage].sort(key=lambda h: h.priority, reverse=False)

    def unregister_hook(self, name: str, stage: Optional[HookStageType] = None) -> None:
        if stage:
            if stage in self._hooks:
                self._hooks[stage] = [h for h in self._hooks[stage] if h.name != name]
        else:
            for s in self._hooks:
                self._hooks[s] = [h for h in self._hooks[s] if h.name != name]

    def enable_hook(self, name: str = None, stage: HookStageType = None, enabled: bool = True) -> None:
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

    def disable_hook(self, name: str = None, stage: HookStageType = None) -> None:
        self.enable_hook(name=name, stage=stage, enabled=False)

    def run_hooks(self, stage: HookStageType, **kwargs) -> Optional[Any]:
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
