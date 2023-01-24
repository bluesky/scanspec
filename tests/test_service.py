from dataclasses import asdict
from typing import Any

import pytest
from fastapi.testclient import TestClient

from scanspec.service import PointsFormat, PointsRequest, app
from scanspec.specs import Line


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# MIDPOINTS TEST(S) #
@pytest.mark.parametrize(
    "format,expected_midpoints",
    [
        (PointsFormat.FLOAT_LIST, [0.0, 0.25, 0.5, 0.75, 1.0]),
        (PointsFormat.STRING, "[0.   0.25 0.5  0.75 1.  ]"),
        (
            PointsFormat.BASE64_ENCODED,
            "AAAAAAAAAAAAAAAAAADQPwAAAAAAAOA/AAAAAAAA6D8AAAAAAADwPw==",
        ),
    ],
    ids=["float_list", "string", "base64"],
)
def test_midpoints(
    client: TestClient, format: PointsFormat, expected_midpoints: Any
) -> None:
    request = PointsRequest(Line("x", 0.0, 1.0, 5), max_frames=5, format=format)
    response = client.post("/midpoints", json=asdict(request))
    assert response.status_code == 200
    assert response.json() == {
        "total_frames": 5,
        "returned_frames": 5,
        "format": format.value,
        "midpoints": {"x": expected_midpoints},
    }


def test_subsampling(client: TestClient) -> None:
    spec = Line("x", 0, 10, 5) * Line("y", 0, 10, 5)
    request = PointsRequest(spec, max_frames=8, format=PointsFormat.FLOAT_LIST)
    response = client.post("/midpoints", json=asdict(request))
    assert response.status_code == 200
    assert response.json() == {
        "total_frames": 25,
        "returned_frames": 8,
        "format": "FLOAT_LIST",
        "midpoints": {"x": [0.0, 0.0, 10.0, 10.0], "y": [0.0, 10.0, 0.0, 10.0]},
    }


# BOUNDS TEST(S) #
@pytest.mark.parametrize(
    "format,expected_lower,expected_upper",
    [
        (
            PointsFormat.FLOAT_LIST,
            [-0.125, 0.125, 0.375, 0.625, 0.875],
            [0.125, 0.375, 0.625, 0.875, 1.125],
        ),
        (
            PointsFormat.STRING,
            "[-0.125  0.125  0.375  0.625  0.875]",
            "[0.125 0.375 0.625 0.875 1.125]",
        ),
        (
            PointsFormat.BASE64_ENCODED,
            "AAAAAAAAwL8AAAAAAADAPwAAAAAAANg/AAAAAAAA5D8AAAAAAADsPw==",
            "AAAAAAAAwD8AAAAAAADYPwAAAAAAAOQ/AAAAAAAA7D8AAAAAAADyPw==",
        ),
    ],
    ids=["float_list", "string", "base64"],
)
def test_bounds(
    client: TestClient, format: PointsFormat, expected_lower: Any, expected_upper: Any
) -> None:
    request = PointsRequest(Line("x", 0.0, 1.0, 5), max_frames=5, format=format)
    response = client.post("/bounds", json=asdict(request))
    assert response.status_code == 200
    assert response.json() == {
        "total_frames": 5,
        "returned_frames": 5,
        "format": format.value,
        "lower": {"x": expected_lower},
        "upper": {"x": expected_upper},
    }


# GAP TEST(S) #
def test_gap(client: TestClient) -> None:
    spec = Line("y", 0.0, 10.0, 3) * Line("x", 0.0, 10.0, 3)
    response = client.post("/gap", json=spec.serialize())
    assert response.status_code == 200
    assert response.json() == {
        "gap": [True, False, False, True, False, False, True, False, False]
    }


# SMALLEST STEP TEST(S) #
def test_smallest_step(client: TestClient) -> None:
    spec = Line("y", 0.0, 10.0, 3) * Line("x", 0.0, 10.0, 5)
    response = client.post("/smalleststep", json=spec.serialize())
    assert response.status_code == 200
    assert response.json() == {"absolute": 2.5, "per_axis": {"y": 5.0, "x": 2.5}}


# VALIDATE SPEC TEST(S) #
def test_validate_spec(client: TestClient) -> None:
    spec = Line.bounded("x", 0, 1, 5)
    response = client.post("/valid", json=spec.serialize())
    assert response.status_code == 200
    assert response.json() == {
        "input_spec": {
            "axis": "x",
            "start": 0.1,
            "stop": 0.9,
            "num": 5,
            "type": "Line",
        },
        "valid_spec": {
            "axis": "x",
            "start": 0.1,
            "stop": 0.9,
            "num": 5,
            "type": "Line",
        },
    }


def test_validate_invalid_spec(client: TestClient) -> None:
    spec = {"type": "Line", "axis": "x", "start": 0.0, "num": 10}
    response = client.post("/valid", json=spec)
    assert response.status_code == 422
