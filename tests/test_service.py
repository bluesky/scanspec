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
