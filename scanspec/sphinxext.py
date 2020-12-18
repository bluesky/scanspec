from inspect import Parameter, signature
from typing import Iterator, List, Optional, Tuple

from matplotlib.sphinxext.plot_directive import PlotDirective
from pydantic import BaseModel
from pydantic.decorator import ValidatedFunction
from pydantic.typing import display_as_type

from scanspec.specs import _modifier


def validated_function_ignores(vd: ValidatedFunction) -> Iterator[str]:
    # Pydantic adds args and kwargs, remove them if not in original func
    parameters = signature(vd.raw_function).parameters
    kinds = {p.kind for p in parameters.values()}
    if Parameter.VAR_POSITIONAL not in kinds:
        yield vd.v_args_name
    if Parameter.VAR_KEYWORD not in kinds:
        yield vd.v_kwargs_name
    # Ignore the first arg as it is the classname
    # This wouldn't work for @staticmethod, but we don't have any
    yield list(parameters)[0]


def get_model_ignores(what: str, obj) -> Tuple[Optional[BaseModel], List[str]]:
    model = None
    ignores = []
    if what == "class" and issubclass(obj, BaseModel):
        # If we are a BaseModel, use this
        model = obj
    elif what == "method" and issubclass(getattr(obj, "model", type), BaseModel):
        # If we have been decorated with validate_arguments, use the model
        model = obj.model
        ignores = list(validated_function_ignores(obj.vd))
    if model:
        for name, field in model.__fields__.items():
            if field.field_info.const and not field.required:
                # Suppress this one as it is probably used like this:
                # https://pydantic-docs.helpmanual.io/usage/types/#literal-type
                ignores.append(name)
    return model, ignores


def process_docstring(app, what, name, obj, options, lines):
    model, ignores = get_model_ignores(what, obj)
    if model:
        # Insert it after the first gap
        try:
            index = lines.index("") + 1
        except ValueError:
            lines.append("")
            index = len(lines)
        # Add types from each field
        for name, field in model.__fields__.items():
            if name not in ignores:
                # UnionModifier expanded this so creation from
                # JSON works, put the original back for docs
                if field.type_ is _modifier.spec_union:
                    type_string = "Spec"
                elif field.type_ is _modifier.region_union:
                    type_string = "Region"
                else:
                    type_string = display_as_type(field.type_)
                description = field.field_info.description
                if type_string == "ConstrainedIntValue":
                    type_string = "int"
                    translate = dict(
                        gt=">", ge=">=", lt="<", le="<=", multiple_of="multiple of"
                    )
                    conditions = []
                    for attr, text in translate.items():
                        v = getattr(field.type_, attr)
                        if v:
                            conditions.append(f"{text} {v}")
                    if conditions:
                        description += f". Must be {' ,'.join(conditions)}"
                lines.insert(
                    index, f":param {type_string} {name}: {description}",
                )
                index += 1
        lines.insert(index, "")


def process_signature(app, what, name, obj, options, signature, return_annotation):
    model, ignores = get_model_ignores(what, obj)
    if model:
        # Recreate signature from the model
        args = []
        for name, field in model.__fields__.items():
            if name not in ignores:
                if field.required:
                    args.append(name)
                else:
                    args.append(f"{name}={field.default!r}")
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
