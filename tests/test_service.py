import graphql
import pytest
from numpy import array

from scanspec.service import Points, schema


# Returns a dummy points dataclass for resolver testing
@pytest.fixture
def points() -> Points:
    return Points(array([1.5, 0.0, 0.25, 1.0, 0.0]))


# VALIDATE GET_POINTS RESOLVER FUNCTIONS
def test_float_list(points):
    assert points.float_list() == [1.5, 0.0, 0.25, 1.0, 0.0]


def test_string(points):
    assert points.string() == "[1.5  0.   0.25 1.   0.  ]"


def test_b64(points):
    assert points.b64() == "AAAAAAAA+D8AAAAAAAAAAAAAAAAAANA/AAAAAAAA8D8AAAAAAAAAAA=="


def test_decodeb64(points):
    assert points.b64Decode() == "[1.5  0.   0.25 1.   0.  ]"


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
def test_get_points_axis():
    query_str = """
{
  getPoints(spec: {Product: {outer: {Line: {key: "x", start: 0, stop: 1, num: 2}},
  inner: {Line: {key: "y", start: 0, stop: 1, num: 3}}}}) {
    axes {
      axis
    }
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": {"axes": [{"axis": "x"}, {"axis": "y"}]}
    }


def test_get_points_lower():
    query_str = """
{
  getPoints(spec: {Product: {outer: {Line: {key: "x", start: 0, stop: 1, num: 2}},
  inner: {Line: {key: "y", start: 0, stop: 1, num: 3}}}}) {
    axes {
      lower{
        floatList
      }
    }
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": {
            "axes": [
                {"lower": {"floatList": [0, 0, 0, 1, 1, 1]}},
                {"lower": {"floatList": [-0.25, 0.25, 0.75, -0.25, 0.25, 0.75]}},
            ]
        }
    }


def test_get_points_middle():
    query_str = """
{
  getPoints(spec: {Product: {outer: {Line: {key: "x", start: 0, stop: 1, num: 2}},
  inner: {Line: {key: "y", start: 0, stop: 1, num: 3}}}}) {
    axes {
      middle{
        floatList
      }
    }
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": {
            "axes": [
                {"middle": {"floatList": [0, 0, 0, 1, 1, 1]}},
                {"middle": {"floatList": [0, 0.5, 1, 0, 0.5, 1]}},
            ]
        }
    }


def test_get_points_upper():
    query_str = """
{
  getPoints(spec: {Product: {outer: {Line: {key: "x", start: 0, stop: 1, num: 2}},
  inner: {Line: {key: "y", start: 0, stop: 1, num: 3}}}}) {
    axes {
      upper{
        floatList
      }
    }
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": {
            "axes": [
                {"upper": {"floatList": [0, 0, 0, 1, 1, 1]}},
                {"upper": {"floatList": [0.25, 0.75, 1.25, 0.25, 0.75, 1.25]}},
            ]
        }
    }


def test_get_points_numPoints():
    query_str = """
{
  getPoints(spec: {Product: {outer: {Line: {key: "x", start: 0, stop: 1, num: 2}}
  inner: {Line: {key: "y", start: 0, stop: 1, num: 3}}}}) {
    numPoints
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": {"numPoints": 6}
    }
