from typing import Any

import strawberry
from fastapi import FastAPI
from resolvers import reduce_frames, validate_spec
from specs import PointsResponse
from strawberry.fastapi import GraphQLRouter

from scanspec.core import Path
from scanspec.specs import Line, Spec

# Here is the manual version of what we are trying to do

# @strawberry.input
# class LineInput(Line): ...


# @strawberry.input
# class ZipInput(Zip): ...


# @strawberry.input(one_of=True)
# class SpecInput:
#     ...

#     line: LineInput | None = strawberry.UNSET
#     zip: ZipInput | None = strawberry.UNSET


def generate_input_class() -> type[Any]:
    # This will be our input class, we're going to fiddle with it
    # throughout this function
    class SpecInput: ...

    # We want to go through all the possible scan specs, this isn't
    # currently possible but can be implemented.
    # Raise an issue for a helper function to get all possible scanspec
    # types.
    for spec_type in Spec.types:
        # We make a strawberry input classs using the scanspec pydantic models
        # This isn't possible because scanspec models are actually pydantic
        # dataclasses. We should have a word with Tom about it and probably
        # raise an issue on strawberry.
        @strawberry.experimental.pydantic.input(all_fields=True, model=spec_type)
        class InputClass: ...

        # Renaming the class to LineInput, ZipInput etc. so the
        # schema looks neater
        InputClass.__name__ = spec_type.__name__ + "Input"

        # Add a field to the class called line, zip etc. and make it
        # strawberry.UNSET
        setattr(SpecInput, spec_type.__name__, strawberry.UNSET)

        # Set the type annotation to line | none, zip | none, etc.
        # Strawberry will read this and graphqlify it.
        SpecInput.__annotations__[spec_type.__name__] = InputClass | None

    # This is just a programtic equivalent of
    # @strawberry.input(one_of=True)
    # class SpecInput:
    #    ...
    return strawberry.input(one_of=True)(SpecInput)


SpecInput = generate_input_class()


@strawberry.type
class Query:
    @strawberry.field
    def validate(self, spec: SpecInput) -> str:
        return validate_spec(spec)

    @strawberry.field
    def get_points(self, spec: Line, max_frames: int | None = 10000) -> PointsResponse:
        """Calculate the frames present in the scan plus some metadata about the points.

        Args:
            spec: The specification of the scan
            max_frames: The maximum number of frames the user wishes to receive
        """

        dims = spec.calculate()  # Grab dimensions from spec

        path = Path(dims)  # Convert to a path

        # TOTAL FRAMES
        total_frames = len(path)  # Capture the total length of the path

        # MAX FRAMES
        # Limit the consumed data by the max_frames argument
        if max_frames and (max_frames < len(path)):
            # Cap the frames by the max limit
            path = reduce_frames(dims, max_frames)
        # WARNING: path object is consumed after this statement
        chunk = path.consume(max_frames)

        return PointsResponse(chunk, total_frames)


schema = strawberry.Schema(Query)

graphql_app = GraphQLRouter(schema, path="/", graphql_ide="apollo-sandbox")

app = FastAPI()
app.include_router(graphql_app, prefix="/graphql")
