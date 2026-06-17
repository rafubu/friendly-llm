from __future__ import annotations

import re
from typing import Any

from .schemas import ModelfileSpec


class ModelfileParseError(Exception):
    pass


_COMMANDS = {"from", "parameter", "system", "template", "message", "adapter", "license"}


def parse_modelfile(content: str) -> ModelfileSpec:
    spec = ModelfileSpec(from_model="", parameters={})

    current_system: list[str] = []
    current_template: list[str] = []
    current_license: list[str] = []
    in_system = False
    in_template = False
    in_license = False

    for line in content.split("\n"):
        stripped = line.strip()

        if in_system:
            if stripped == '"""' or stripped == "'''":
                spec.system = "\n".join(current_system).strip()
                current_system = []
                in_system = False
            elif stripped.endswith('"""') or stripped.endswith("'''"):
                current_system.append(stripped[:-3].rstrip())
                spec.system = "\n".join(current_system).strip()
                current_system = []
                in_system = False
            else:
                current_system.append(line)
            continue

        if in_template:
            if stripped == '"""' or stripped == "'''":
                spec.template = "\n".join(current_template).strip()
                current_template = []
                in_template = False
            elif stripped.endswith('"""') or stripped.endswith("'''"):
                current_template.append(stripped[:-3].rstrip())
                spec.template = "\n".join(current_template).strip()
                current_template = []
                in_template = False
            else:
                current_template.append(line)
            continue

        if in_license:
            if stripped == '"""' or stripped == "'''":
                spec.license = "\n".join(current_license).strip()
                current_license = []
                in_license = False
            elif stripped.endswith('"""') or stripped.endswith("'''"):
                current_license.append(stripped[:-3].rstrip())
                spec.license = "\n".join(current_license).strip()
                current_license = []
                in_license = False
            else:
                current_license.append(line)
            continue

        if not stripped or stripped.startswith("#"):
            continue

        if not re.match(r"^\w+", stripped):
            raise ModelfileParseError(f"Expected command at line: {stripped!r}")

        cmd_match = re.match(r"^(\w+)\s+(.*)", stripped)
        if not cmd_match:
            raise ModelfileParseError(f"Invalid command syntax: {stripped!r}")

        cmd = cmd_match.group(1).lower()
        rest = cmd_match.group(2)

        if cmd not in _COMMANDS:
            raise ModelfileParseError(f"Unknown command {cmd!r} in Modelfile")

        if cmd == "from":
            spec.from_model = rest.strip()

        elif cmd == "parameter":
            param_match = re.match(r"^(\S+)\s+(.+)", rest)
            if param_match:
                key = param_match.group(1).lower()
                val = _parse_parameter_value(param_match.group(2))
                spec.parameters[key] = val

        elif cmd == "system":
            if rest.startswith('"""') or rest.startswith("'''"):
                delim = rest[:3]
                inner = rest[3:]
                if inner.endswith(delim):
                    spec.system = inner[:-3].strip()
                else:
                    current_system.append(inner)
                    in_system = True
            else:
                spec.system = rest.strip()

        elif cmd == "template":
            if rest.startswith('"""') or rest.startswith("'''"):
                delim = rest[:3]
                inner = rest[3:]
                if inner.endswith(delim):
                    spec.template = inner[:-3].strip()
                else:
                    current_template.append(inner)
                    in_template = True
            else:
                spec.template = rest.strip()

        elif cmd == "message":
            msg_match = re.match(r"(system|user|assistant|tool)\s+(.*)", rest, re.DOTALL)
            if msg_match:
                role = msg_match.group(1)
                content = msg_match.group(2).strip()
                spec.messages.append({"role": role, "content": content})

        elif cmd == "license":
            if rest.startswith('"""') or rest.startswith("'''"):
                delim = rest[:3]
                inner = rest[3:]
                if inner.endswith(delim):
                    spec.license = inner[:-3].strip()
                else:
                    current_license.append(inner)
                    in_license = True
            else:
                spec.license = rest.strip()

        elif cmd == "adapter":
            spec.adapter = rest.strip()

    return spec


def _parse_parameter_value(val: str) -> Any:
    val = val.strip()
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


def _escape_modelfile_value(text: str) -> str:
    return text.replace('"""', '\\"\\"\\"')


def generate_modelfile_string(spec: ModelfileSpec) -> str:
    lines = [f"FROM {spec.from_model}"]
    for key, val in spec.parameters.items():
        lines.append(f"PARAMETER {key} {val}")
    if spec.system:
        lines.append(f'SYSTEM """{_escape_modelfile_value(spec.system)}"""')
    if spec.template:
        lines.append(f'TEMPLATE """{_escape_modelfile_value(spec.template)}"""')
    for msg in spec.messages:
        lines.append(f'MESSAGE {msg["role"]} {msg["content"]}')
    if spec.license:
        lines.append(f'LICENSE """{_escape_modelfile_value(spec.license)}"""')
    if spec.adapter:
        lines.append(f"ADAPTER {spec.adapter}")
    return "\n".join(lines)
