from app.commands.audit import register as register_audit
from app.commands.benchmark import register as register_benchmark
from app.commands.doctor import register as register_doctor
from app.commands.merge import register as register_merge
from app.commands.test import register as register_test

from app.commands.validate import register as register_validate

__all__ = [
    "register_audit",
    "register_benchmark",
    "register_doctor",
    "register_merge",
    "register_test",
    "register_validate",
]
