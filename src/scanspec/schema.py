import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter


def get_lines():
    # example line
    return [Line(name="line", axis="x", start=0.2, stop=0.3, num=4)]


@strawberry.input
class Line:
    name: str
    axis: str = strawberry.field(description="An identifier for what to move")
    start: float = strawberry.field(
        description="Midpoint of the first point of the line"
    )
    stop: float = strawberry.field(description="Midpoint of the last point of the line")
    num: int = strawberry.field(description="Number of frames to produce")


@strawberry.type
class Query:
    @strawberry.field
    def validate(self, spec: Line) -> str:
        return "accepted"


schema = strawberry.Schema(Query)

graphql_app = GraphQLRouter(schema, path="/", graphql_ide="apollo-sandbox")

app = FastAPI()
app.include_router(graphql_app, prefix="/graphql")
