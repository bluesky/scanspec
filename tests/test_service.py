import base64
from typing import Any, Mapping

import graphql
import numpy as np
import pytest
from graphql.error.graphql_error import GraphQLError
from graphql.type.schema import GraphQLSchema, assert_schema

from scanspec.service import Points, scanspec_schema
from scanspec.specs import Line


# Returns a dummy 'points' dataclass for resolver testing
@pytest.fixture
def points() -> Points:
    return Points(np.array([1.5, 0.0, 0.25, 1.0, 0.0]))


# GET_POINTS RESOLVER TEST(S) #
def test_float_list(points) -> None:
    assert points.float_list() == [1.5, 0.0, 0.25, 1.0, 0.0]


def test_string(points) -> None:
    assert points.string() == "[1.5  0.   0.25 1.   0.  ]"


def test_b64(points) -> None:
    assert points.b64() == "AAAAAAAA+D8AAAAAAAAAAAAAAAAAANA/AAAAAAAA8D8AAAAAAAAAAA=="


def test_decodeb64(points) -> None:
    encoded_points = points.b64()
    s = base64.decodebytes(encoded_points.encode())
    t = np.frombuffer(s, dtype=np.float64)
    assert np.array2string(t) == "[1.5  0.   0.25 1.   0.  ]"


# VALIDATE SPEC QUERY TEST(S) #
def test_validate_spec() -> None:
    spec = Line.bounded("x", 0, 1, 5)
    query_str = (
        """
{
    validateSpec(spec: %s)
}
    """
        % spec.to_gql_input()
    )
    assert graphql_exec(scanspec_schema, query_str) == {
        "validateSpec": {"Line": {"axis": "x", "start": 0.1, "stop": 0.9, "num": 5}}
    }


# GET POINTS QUERY TEST(S) #
GRID = Line("x", 0, 1, 2) * Line("y", 0, 1, 3)


def test_get_points_axis() -> None:

    query_str = (
        """
{
  getPoints(spec: %s) {
    axes {
      axis
    }
  }
}
    """
        % GRID.to_gql_input()
    )
    assert graphql_exec(scanspec_schema, query_str) == {
        "getPoints": {"axes": [{"axis": "x"}, {"axis": "y"}]}
    }


def test_get_points_lower() -> None:
    query_str = (
        """
{
  getPoints(spec: %s) {
    axes {
      lower{
        floatList
      }
    }
  }
}
    """
        % GRID.to_gql_input()
    )
    assert graphql_exec(scanspec_schema, query_str) == {
        "getPoints": {
            "axes": [
                {"lower": {"floatList": [0, 0, 0, 1, 1, 1]}},
                {"lower": {"floatList": [-0.25, 0.25, 0.75, -0.25, 0.25, 0.75]}},
            ]
        }
    }


def test_get_points_midpoints() -> None:
    query_str = (
        """
{
  getPoints(spec: %s) {
    axes {
      midpoints{
        floatList
      }
    }
  }
}
    """
        % GRID.to_gql_input()
    )
    assert graphql_exec(scanspec_schema, query_str) == {
        "getPoints": {
            "axes": [
                {"midpoints": {"floatList": [0, 0, 0, 1, 1, 1]}},
                {"midpoints": {"floatList": [0, 0.5, 1, 0, 0.5, 1]}},
            ]
        }
    }


def test_get_points_upper() -> None:
    query_str = (
        """
{
  getPoints(spec: %s) {
    axes {
      upper{
        floatList
      }
    }
  }
}
    """
        % GRID.to_gql_input()
    )
    assert graphql_exec(scanspec_schema, query_str) == {
        "getPoints": {
            "axes": [
                {"upper": {"floatList": [0, 0, 0, 1, 1, 1]}},
                {"upper": {"floatList": [0.25, 0.75, 1.25, 0.25, 0.75, 1.25]}},
            ]
        }
    }


def test_get_points_upper_limited() -> None:
    spec = Line("x", 0, 10, 5) * Line("y", 0, 10, 5)
    query_str = (
        """
{
  getPoints(spec: %s, maxFrames: 8) {
    totalFrames
    returnedFrames
    axes {
      upper{
        floatList
      }
    }
  }
}
    """
        % spec.to_gql_input()
    )
    assert graphql_exec(scanspec_schema, query_str) == {
        "getPoints": {
            "totalFrames": 25,
            "returnedFrames": 4,
            "axes": [
                {"upper": {"floatList": [0, 0, 10, 10]}},
                {"upper": {"floatList": [1.25, 11.25, 1.25, 11.25]}},
            ],
        }
    }


def test_get_points_smallest_step() -> None:
    query_str = (
        """
{
  getPoints(spec: %s) {
    axes {
      axis
      smallestStep
    }
  }
}

    """
        % GRID.to_gql_input()
    )
    assert graphql_exec(scanspec_schema, query_str) == {
        "getPoints": {
            "axes": [
                {"axis": "x", "smallestStep": 0},
                {"axis": "y", "smallestStep": 0.5},
            ]
        }
    }


def test_get_points_total_frames() -> None:
    query_str = (
        """
{
  getPoints(spec: %s) {
    totalFrames
  }
}
    """
        % GRID.to_gql_input()
    )
    assert graphql_exec(scanspec_schema, query_str) == {"getPoints": {"totalFrames": 6}}


def test_get_points_abs_smallest_step() -> None:
    spec = Line("x", 0, 10, 3) * Line("y", 0, 10, 3)
    query_str = (
        """
{
  getPoints(spec: %s) {
    smallestAbsStep
    axes {
      midpoints {
        floatList
      }
    }
  }
}
    """
        % spec.to_gql_input()
    )
    assert graphql_exec(scanspec_schema, query_str) == {
        "getPoints": {
            "smallestAbsStep": 5,
            "axes": [
                {"midpoints": {"floatList": [0, 0, 0, 5, 5, 5, 10, 10, 10]}},
                {"midpoints": {"floatList": [0, 5, 10, 0, 5, 10, 0, 5, 10]}},
            ],
        }
    }


def graphql_exec(schema: GraphQLSchema, query: str) -> Mapping[str, Any]:
    execution_result = graphql.graphql_sync(schema, query)
    if execution_result.errors:
        raise GraphQLError(
            f"Errors found during GraphQL execution: {execution_result.errors}"
        )
    elif not execution_result.data:
        raise GraphQLError("No data or errors returned from query")
    else:
        return execution_result.data


def test_schema() -> None:
    assert_schema(scanspec_schema)
