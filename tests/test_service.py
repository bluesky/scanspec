import graphql

from scanspec.service import schema


def test_validate_spec():
    query_str = """
{
    validateSpec(spec: {BoundedLine: {key: "x", lower: 0, upper: 1, num: 5}})
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "validateSpec": {"Line": {"key": "x", "start": 0.1, "stop": 0.9, "num": 5}}
    }


def test_get_points():
    query_str = """
{
    getPoints(spec: {Product: {outer: {Line: {key: "x", start: 0, stop: 1, num: 5}}
    inner: {Line: {key: "y", start: 0, stop: 1, num: 5}}}})
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": [{"x": [0, 0.25, 0.5, 0.75, 1]}, {"y": [0, 0.25, 0.5, 0.75, 1]}]
    }
