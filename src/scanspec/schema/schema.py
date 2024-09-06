import strawberry
from fastapi import FastAPI
from resolvers import reduce_frames, validate_spec
from specs import Line, PointsResponse
from strawberry.fastapi import GraphQLRouter

from scanspec.core import Path


@strawberry.type
class Query:
    @strawberry.field
    def validate(self, spec: Line) -> str:
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
