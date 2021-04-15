import graphql

from scanspec.service import schema


# VALIDATE SPEC QUERY TEST(S)
def test_validate_spec():
    query_str = """
{
    validateSpec(spec: {BoundedLine: {key: "x", lower: 0, upper: 1, num: 5}})
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "validateSpec": {"Line": {"key": "x", "start": 0.1, "stop": 0.9, "num": 5}}
    }


# GET POINTS QUERY TEST(S)
def test_get_points_key():
    query_str = """
{
  getPoints(spec: {Product: {outer: {Line: {key: "x", start: 0, stop: 1, num: 2}},
  inner: {Line: {key: "y", start: 0, stop: 1, num: 3}}}}) {
    points{
      key
    }
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": {"points": [{"key": "x"}, {"key": "y"}]}
    }


def test_get_points_lower():
    query_str = """
{
  getPoints(spec: {Product: {outer: {Line: {key: "x", start: 0, stop: 1, num: 2}},
  inner: {Line: {key: "y", start: 0, stop: 1, num: 3}}}}) {
    points {
      lower{
        string
        floatList
      }
    }
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": {
            "points": [
                {
                    "lower": {
                        "string": "[0. 0. 0. 1. 1. 1.]",
                        "floatList": [0, 0, 0, 1, 1, 1],
                    }
                },
                {
                    "lower": {
                        "string": "[-0.25  0.25  0.75 -0.25  0.25  0.75]",
                        "floatList": [-0.25, 0.25, 0.75, -0.25, 0.25, 0.75],
                    }
                },
            ]
        }
    }


def test_get_points_middle():
    query_str = """
{
  getPoints(spec: {Product: {outer: {Line: {key: "x", start: 0, stop: 1, num: 2}},
  inner: {Line: {key: "y", start: 0, stop: 1, num: 3}}}}) {
    points {
      middle{
        string
        floatList
      }
    }
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": {
            "points": [
                {
                    "middle": {
                        "string": "[0. 0. 0. 1. 1. 1.]",
                        "floatList": [0, 0, 0, 1, 1, 1],
                    }
                },
                {
                    "middle": {
                        "string": "[0.  0.5 1.  0.  0.5 1. ]",
                        "floatList": [0, 0.5, 1, 0, 0.5, 1],
                    }
                },
            ]
        }
    }


def test_get_points_upper():
    query_str = """
{
  getPoints(spec: {Product: {outer: {Line: {key: "x", start: 0, stop: 1, num: 2}},
  inner: {Line: {key: "y", start: 0, stop: 1, num: 3}}}}) {
    points {
      upper{
        string
        floatList
      }
    }
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": {
            "points": [
                {
                    "upper": {
                        "string": "[0. 0. 0. 1. 1. 1.]",
                        "floatList": [0, 0, 0, 1, 1, 1],
                    }
                },
                {
                    "upper": {
                        "string": "[0.25 0.75 1.25 0.25 0.75 1.25]",
                        "floatList": [0.25, 0.75, 1.25, 0.25, 0.75, 1.25],
                    }
                },
            ]
        }
    }


def test_get_points_numPoints():
    query_str = """
{
  getPoints(spec: {Product: {outer: {Line: {key: "x", start: 0, stop: 1, num: 2}},
  inner: {Line: {key: "y", start: 0, stop: 1, num: 3}}}}) {
    numPoints
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": {"numPoints": 6}
    }
