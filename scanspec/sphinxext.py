from dataclasses import _FIELD_CLASSVAR, Field, is_dataclass  # type: ignore
from typing import Any, Dict, Optional, Tuple

from apischema.json_schema.schema import Schema
from apischema.metadata.keys import SCHEMA_METADATA
from matplotlib.sphinxext.plot_directive import PlotDirective
from typing_extensions import Annotated, get_args, get_origin


def get_type(meta: Tuple[Any]) -> str:
    try:
        field_type = meta[0].__name__
    except AttributeError:
        field_type = meta[0]._name
    return field_type


def get_dataclass_fields(obj: Any) -> Dict[str, Field]:
    fields = obj.__dataclass_fields__
    # Remove class variables
    fields = {
        field_name: field
        for field_name, field in fields.items()
        if field._field_type != _FIELD_CLASSVAR
    }
    return fields


def get_metadata(what: str, obj) -> Optional[Dict[str, Any]]:
    metadata = None
    if what == "class" and is_dataclass(obj):
        # If we are a dataclass, use this
        fields = get_dataclass_fields(obj)
        metadata = get_metadata_from_fields(fields)
    elif what == "method":
        annotations = obj.__annotations__
        # Don't include the return value
        annotations = {
            name: annotation
            for name, annotation in annotations.items()
            if name != "return"
        }
        metadata = get_metadata_from_annotations(annotations)
    return metadata


def get_metadata_from_fields(fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    metadata = {}
    for name, field in fields.items():
        metadata[name] = (field.type, field.metadata)
    return metadata or None


def get_metadata_from_annotations(
    annotations: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    metadata = {}
    for name, annotation in annotations.items():
        if get_origin(annotation) == Annotated:
            metadata[name] = get_args(annotation)
    return metadata or None


def get_description(meta: Any) -> str:
    description = ""
    additional = []
    for arg in meta[1:]:
        arg = arg.get(SCHEMA_METADATA, arg)
        if type(arg) == Schema:
            field_schema = arg.as_dict()
            for key, value in field_schema.items():
                if key == "description":
                    description = value
                else:
                    additional.append(f"{str(key)}: {str(value)}")
    if additional:
        description = description + " - " + ", ".join(additional)
    return description


def process_docstring(app, what, name, obj, options, lines):
    # Must also work for the alternative constructors such as bounded!!
    metadata = get_metadata(what, obj)
    if metadata:
        try:
            index = lines.index("") + 1
        except ValueError:
            lines.append("")
            index = len(lines)
        # Add types from each field
        for name, meta in metadata.items():
            description = get_description(meta)
            field_type = get_type(meta)
            lines.insert(
                index, f":param {field_type} {name}: {description}",
            )
            index += 1
        lines.insert(index, "")


def process_signature(app, what, name, obj, options, signature, return_annotation):
    # Recreate signature from the model
    if is_dataclass(obj):
        fields = get_dataclass_fields(obj)
        args = fields.keys()
        signature = f'({", ".join(args)})'
        return signature, return_annotation


class ExampleSpecDirective(PlotDirective):
    def run(self):
        self.content = (
            ["# Example Spec", "", "from scanspec.plot import plot_spec"]
            + [str(x) for x in self.content]
            + ["plot_spec(spec)"]
        )
        return super().run()


def setup(app):
    app.add_directive("example_spec", ExampleSpecDirective)
    app.connect("autodoc-process-signature", process_signature)
    app.connect("autodoc-process-docstring", process_docstring)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
