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
  getPoints(spec: {Product: {outer: {Line: {key: "x", start: 0, stop: 1, num: 2}},
                            inner: {Line: {key: "y", start: 0, stop: 1, num: 3}}}}){
    key
    upper
    middle
    lower
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": [
            {
                "key": "x",
                "lower": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                "middle": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                "upper": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            },
            {
                "key": "y",
                "lower": [-0.25, 0.25, 0.75, -0.25, 0.25, 0.75],
                "middle": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
                "upper": [0.25, 0.75, 1.25, 0.25, 0.75, 1.25],
            },
        ]
    }
