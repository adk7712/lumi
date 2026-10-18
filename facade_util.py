"""Helper for thin facade modules that re-export names from implementation submodules."""
import sys
import types


def install_facade(facade_name, submodules, live=()):
    """Make ``facade_name`` forward attribute assignment to the submodules that define the name.

    Tests (and callers) monkeypatch ``facade.NAME``; without forwarding, the override would only
    change the facade's copy and the implementation would keep using the original. Names in
    ``live`` are not stored on the facade at all: reads resolve live from the owning submodule.
    """
    subs = tuple(submodules)
    live = tuple(live)

    class _FacadeModule(types.ModuleType):
        def __getattr__(self, name):  # reached only for names absent from the facade dict
            for m in subs:
                if name in m.__dict__:
                    return m.__dict__[name]
            raise AttributeError(f"module {facade_name!r} has no attribute {name!r}")

        def __setattr__(self, name, value):
            owned = False
            for m in subs:
                if name in m.__dict__:
                    setattr(m, name, value)
                    owned = True
            if not owned or name not in live:
                super().__setattr__(name, value)

    mod = sys.modules[facade_name]
    for n in live:
        mod.__dict__.pop(n, None)
    mod.__class__ = _FacadeModule
