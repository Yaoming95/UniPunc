
class ArchitectureRegister(object):
    def __init__(self):
        self.caches = {}

    def register(self, name, cls):
        if name in self.caches:
            raise ValueError(f"{name} has been set.")

        def _register_x_cls(f):
            self.caches[name] = (cls, f)
            return f
        return _register_x_cls

    def choices(self):
        return list(self.caches.keys())

    def build(self, name, args, task):
        cls, f = self.caches[name]
        args = f(args)
        return cls.build_model(args, task)

    def all_names(self):
        return list(self.caches.keys())


pretrain_model_register = ArchitectureRegister()
header_register = ArchitectureRegister()
duration_header_register = ArchitectureRegister()



# This line should be after all others
from .UniPunc import *
