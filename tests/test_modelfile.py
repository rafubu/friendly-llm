from __future__ import annotations

from litert_ollama.modelfile import parse_modelfile, ModelfileParseError, generate_modelfile_string
from litert_ollama.schemas import ModelfileSpec


def test_parse_modelfile_basic():
    content = (
        'FROM gemma4-12b\n'
        'PARAMETER temperature 0.7\n'
        'PARAMETER top_p 0.9\n'
        'SYSTEM """You are a helpful assistant."""\n'
        'LICENSE """MIT"""\n'
    )
    spec = parse_modelfile(content)
    assert spec.from_model == "gemma4-12b"
    assert spec.parameters["temperature"] == 0.7
    assert spec.parameters["top_p"] == 0.9
    assert spec.system == "You are a helpful assistant."
    assert spec.license == "MIT"


def test_parse_modelfile_parameters():
    content = (
        'FROM gemma4-12b\n'
        'PARAMETER temperature 0.5\n'
        'PARAMETER top_k 40\n'
        'PARAMETER seed 42\n'
        'PARAMETER num_ctx 8192\n'
    )
    spec = parse_modelfile(content)
    assert spec.parameters["temperature"] == 0.5
    assert spec.parameters["top_k"] == 40
    assert spec.parameters["seed"] == 42
    assert spec.parameters["num_ctx"] == 8192


def test_parse_modelfile_messages():
    content = (
        'FROM gemma4-12b\n'
        'MESSAGE system You are a math tutor\n'
        'MESSAGE user What is 2+2?\n'
        'MESSAGE assistant The answer is 4\n'
    )
    spec = parse_modelfile(content)
    assert len(spec.messages) == 3
    assert spec.messages[0] == {"role": "system", "content": "You are a math tutor"}
    assert spec.messages[1] == {"role": "user", "content": "What is 2+2?"}
    assert spec.messages[2] == {"role": "assistant", "content": "The answer is 4"}


def test_parse_modelfile_multiline_system():
    content = (
        'FROM gemma4-12b\n'
        'SYSTEM """You are a multilingual\n'
        'coding assistant that speaks\n'
        'Spanish and English."""\n'
    )
    spec = parse_modelfile(content)
    assert "multilingual" in spec.system
    assert "Spanish" in spec.system
    assert "English" in spec.system


def test_parse_modelfile_invalid_command():
    content = "FROM gemma4-12b\nINVALID some text\n"
    try:
        parse_modelfile(content)
        assert False, "Should have raised ModelfileParseError"
    except ModelfileParseError:
        pass


def test_parse_modelfile_no_from():
    content = "PARAMETER temperature 0.7\n"
    spec = parse_modelfile(content)
    assert spec.from_model == ""


def test_generate_modelfile_string():
    spec = ModelfileSpec(
        from_model="gemma4-12b",
        parameters={"temperature": 0.7, "top_p": 0.9},
        system="You are helpful",
        messages=[{"role": "user", "content": "Hello"}],
    )
    result = generate_modelfile_string(spec)
    assert "FROM gemma4-12b" in result
    assert "PARAMETER temperature 0.7" in result
    assert "SYSTEM" in result
    assert "MESSAGE" in result
