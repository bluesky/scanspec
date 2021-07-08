from scanspec.specs import Line, Static
from scanspec.sphinxext import process_docstring, process_signature


def test_signature_static():
    sig = process_signature(
        "app", "class", "name", Static, "options", "signature", "return_annotation"
    )
    assert sig == ("(axis, value, num=1)", "return_annotation")


def test_docstring_line():
    lines = ["Here's some", "lines", "", "And a bit", "more"]
    process_docstring("app", "class", "name", Line, "options", lines)
    assert lines == [
        "Here's some",
        "lines",
        "",
        ":param str axis: An identifier for what to move",
        ":param float start: Midpoint of the first point of the line",
        ":param float stop: Midpoint of the last point of the line",
        ":param int num: Number of points to produce - minimum: 1",
        "",
        "And a bit",
        "more",
    ]


def test_docstring_line_bounded():
    lines = ["Here's some", "lines"]
    process_docstring("app", "method", "name", Line.bounded, "options", lines)
    assert lines == [
        "Here's some",
        "lines",
        "",
        ":param str axis: An identifier for what to move",
        ":param float lower: Lower bound of the first point of the line",
        ":param float upper: Upper bound of the last point of the line",
        ":param int num: Number of points to produce - minimum: 1",
        "",
    ]
