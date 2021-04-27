import graphql
import pytest
from numpy import array

from scanspec.service import Bit, schema


# Returns a dummy bit dataclass for resolver testing
@pytest.fixture
def bit() -> Bit:
    return Bit(array([1.5, 0.0, 0.25, 1.0, 0.0]))


# VALIDATE GET_POINTS RESOLVER FUNCTIONS
def test_float_list(bit):
    assert bit.float_list() == [1.5, 0.0, 0.25, 1.0, 0.0]


def test_string(bit):
    assert bit.string() == "[1.5  0.   0.25 1.   0.  ]"


def test_b64(bit):
    assert bit.b64() == "AAAAAAAA+D8AAAAAAAAAAAAAAAAAANA/AAAAAAAA8D8AAAAAAAAAAA=="


def test_decodeb64(bit):
    assert bit.b64Decode() == "[1.5  0.   0.25 1.   0.  ]"


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
        floatList
      }
    }
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": {
            "points": [
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
    points {
      middle{
        floatList
      }
    }
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": {
            "points": [
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
    points {
      upper{
        floatList
      }
    }
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": {
            "points": [
                {"upper": {"floatList": [0, 0, 0, 1, 1, 1]}},
                {"upper": {"floatList": [0.25, 0.75, 1.25, 0.25, 0.75, 1.25]}},
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
