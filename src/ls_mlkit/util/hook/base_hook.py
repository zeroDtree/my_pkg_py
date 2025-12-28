from enum import Enum
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

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


class HookHandler(Generic[HookStageType]):

    def __init__(self, manager: "HookManager[HookStageType]", hook: Hook[HookStageType]):
        self._manager = manager
        self._hook = hook

    @property
    def hook(self) -> Hook[HookStageType]:
        return self._hook

    def enable(self) -> None:
        self._hook.enabled = True

    def disable(self) -> None:
        self._hook.enabled = False

    def remove(self) -> None:
        self._manager.unregister_hook(name=self._hook.name, stage=self._hook.stage)

    def __repr__(self):
        state = "enabled" if self._hook.enabled else "disabled"
        return f"<HookHandler {self._hook.name} ({state})>"


class HookManager(Generic[HookStageType]):

    def __init__(self) -> None:
        self._hooks: Dict[HookStageType, list[Hook[HookStageType]]] = {}

    def register_hook(self, hook: Hook[HookStageType]) -> HookHandler[HookStageType]:
        self._hooks.setdefault(hook.stage, []).append(hook)
        self._hooks[hook.stage].sort(key=lambda h: h.priority, reverse=False)
        return HookHandler(self, hook=hook)

    def register_hooks(self, hooks: list[Hook[HookStageType]]) -> list[HookHandler[HookStageType]]:
        return [self.register_hook(hook) for hook in hooks]

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

    def run_hooks(self, stage: HookStageType, tgt_key_name=None, **kwargs) -> Optional[Any]:
        """Executes all enabled hooks for a given stage, optionally updating or collecting results in kwargs,
        and returns either the final modified kwargs or a specific key's value.

        Args:
            stage (``HookStageType``): _description_
            tgt_key_name (``_type_``, *optional*): target key name. Defaults to None.
        """
        hook_output = None

        if stage is not None and stage in self._hooks:
            for hook in self._hooks[stage]:
                if not hook.enabled:
                    continue
                hook_output = hook(**kwargs)
                if tgt_key_name is not None:
                    kwargs[tgt_key_name] = hook_output
                else:
                    kwargs = hook_output

        if tgt_key_name is not None:
            return kwargs[tgt_key_name]
        elif tgt_key_name is None:
            return kwargs

    def list_hooks(self) -> None:
        for stage, hooks in self._hooks.items():
            print(f"[{stage}]")
            for h in hooks:
                print(f"  - {h} {'(enabled)' if h.enabled else '(disabled)'}")
