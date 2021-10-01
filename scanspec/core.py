from types import new_class
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

import numpy as np
from apischema import deserialize, deserializer, serialize, serializer, type_name
from apischema.conversions import Conversion
from apischema.metadata import conversion
from apischema.objects import object_deserialization
from apischema.tagged_unions import Tagged, TaggedUnion, get_tagged
from apischema.utils import to_pascal_case

__all__ = [
    "alternative_constructor",
    "Serializable",
    "AxesPoints",
    "if_instance_do",
    "Frames",
    "gap_between_frames",
    "squash_frames",
    "Path",
    "Midpoints",
]


def _rec_subclasses(cls: type) -> Iterator[type]:
    """Recursive implementation of type.__subclasses__"""
    for sub_cls in cls.__subclasses__():
        yield sub_cls
        yield from _rec_subclasses(sub_cls)


# {cls_name: [functions]}
_alternative_constructors: Dict[str, List[Callable]] = {}

if TYPE_CHECKING:
    # Close enough for mypy
    alternative_constructor = staticmethod
else:

    def alternative_constructor(f):
        """Register an alternative constructor for this class. This will be returned
        as a staticmethod so the signature should not include self/cls.

        >>> import dataclasses
        >>> @dataclasses.dataclass
        ... class Foo:
        ...     a: int
        ...     @alternative_constructor
        ...     def doubled(b: int) -> "Foo":
        ...         return Foo(b * 2)
        ...
        >>> Foo.doubled(2)
        Foo(a=4)
        """
        cls_name = f.__qualname__.split(".")[0]
        _alternative_constructors.setdefault(cls_name, []).append(f)
        return staticmethod(f)


generic_name = type_name(lambda cls, *args: cls.__name__)


def _as_tagged_union(cls: Type):
    params = tuple(getattr(cls, "__parameters__", ()))
    tagged_union_bases: tuple = (TaggedUnion,)
    if params:
        tagged_union_bases = (TaggedUnion, Generic[params])
        generic_name(cls)
        prev_init_subclass = getattr(cls, "__init_subclass__", None)

        def __init_subclass__(cls, **kwargs):
            if prev_init_subclass is not None:
                prev_init_subclass(**kwargs)
            generic_name(cls)

        cls.__init_subclass__ = classmethod(__init_subclass__)

    def with_params(cls: type) -> Any:
        return cls[params] if params else cls

    def serialization() -> Conversion:
        annotations = {
            # Assume that subclasses have same generic parameters than cls
            sub.__name__: Tagged[with_params(sub)]
            for sub in _rec_subclasses(cls)
        }
        namespace = {"__annotations__": annotations}
        tagged_union = new_class(
            cls.__name__, tagged_union_bases, exec_body=lambda ns: ns.update(namespace)
        )
        return Conversion(
            lambda obj: tagged_union(**{obj.__class__.__name__: obj}),
            source=with_params(cls),
            target=with_params(tagged_union),
            # Conversion must not be inherited because it would lead to
            # infinite recursion otherwise
            inherited=False,
        )

    def deserialization() -> Conversion:
        annotations: dict[str, Any] = {}
        namespace: dict[str, Any] = {"__annotations__": annotations}
        for sub in _rec_subclasses(cls):
            # Assume that subclasses have same generic parameters than cls
            annotations[sub.__name__] = Tagged[with_params(sub)]
            # Add tagged fields for all its alternative constructors
            for constructor in _alternative_constructors.get(sub, ()):
                # Build the alias of the field
                alias = to_pascal_case(constructor.__name__)
                # object_deserialization uses get_type_hints, but the constructor
                # return type is stringified and the class not defined yet,
                # so it must be assigned manually
                constructor.__annotations__["return"] = with_params(sub)
                # Use object_deserialization to wrap constructor as deserializer
                deserialization = object_deserialization(constructor, generic_name)
                # Add constructor tagged field with its conversion
                annotations[alias] = Tagged[with_params(sub)]
                namespace[alias] = Tagged(conversion(deserialization=deserialization))
        # Create the deserialization tagged union class
        tagged_union = new_class(
            cls.__name__, tagged_union_bases, exec_body=lambda ns: ns.update(namespace)
        )
        return Conversion(
            lambda obj: get_tagged(obj)[1],
            source=with_params(tagged_union),
            target=with_params(cls),
        )

    setattr(cls, "__is_tagged_union__", True)
    deserializer(lazy=deserialization, target=cls)
    serializer(lazy=serialization, source=cls)
    return cls


#: A subclass of `Serializable`
S = TypeVar("S", bound="Serializable")


class Serializable:
    """Base class for registering apischema (de)serialization conversions.
    Each direct subclass will be registered for (de)serialization as a tagged union
    of its subclasses, using the pattern documented here:
    https://wyfo.github.io/apischema/examples/subclass_tagged_union/"""

    # Base class which will directly inherit from Serializable
    _base_serializable: ClassVar[Type["Serializable"]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if Serializable in cls.__bases__ and not hasattr(cls, "__is_tagged_union__"):
            cls._base_serializable = cls
            _as_tagged_union(cls)

    def serialize(self) -> Mapping[str, Any]:
        """Serialize to a dictionary representation"""
        # Base serializable class must be passed to serialize in order to use its
        # registered conversion (which is not inherited)
        return serialize(self._base_serializable, self)

    @classmethod
    def deserialize(cls: Type[S], serialization: Mapping[str, Any]) -> S:
        """Deserialize from a dictionary representation"""
        inst = deserialize(cls._base_serializable, serialization)
        assert isinstance(inst, cls)
        return inst


#: Type of an axis
K = TypeVar("K")

#: Map of axes to points_ndarray
#: E.g. {xmotor: array([0, 1, 2]), ymotor: array([2, 2, 2])}
AxesPoints = Dict[K, np.ndarray]


def if_instance_do(x, cls: Type, func: Callable):
    """If x is of type cls then return func(x), otherwise return NotImplemented.
    Used as a helper when implementing operator overloading"""
    if isinstance(x, cls):
        return func(x)
    else:
        return NotImplemented


#: A subclass of `Frames`
F = TypeVar("F", bound="Frames")


class Frames(Generic[K]):
    """Represents a series of scan frames along a number of axes. During a scan
    each axis will traverse lower-midpoint-upper for each frame.

    Args:
        midpoints: The centre points of the frames for each axis
        lower: Lower bounds if different from midpoints
        upper: Upper bounds if different from midpoints
        gap: If supplied, define if there is a gap between frame and previous
            otherwise it is calculated by looking at lower and upper bounds

    Typically used in two ways:

    - A list of Frames returned from `Spec.calculate` represents a scan as a
      linear stack of frames. Interpreted as nested from slowest moving to
      fastest moving, so each faster Frames object will iterate once per
      position of the slower Frames. It is passed to a `Path` for calculation
      of the actual scan path.
    - A single Frames object returned from `Path.consume` represents a chunk
      of frames forming part of a scan path, for interpretation by the code
      that will actually perform the scan.

    See Also:
        `technical-terms`
    """

    def __init__(
        self,
        midpoints: AxesPoints[K],
        lower: AxesPoints[K] = None,
        upper: AxesPoints[K] = None,
        gap: np.ndarray = None,
    ):
        #: The centre points of the scan for each axis
        self.midpoints = midpoints
        #: The lower bounds of each scan point in each axis for fly-scanning
        self.lower = lower or midpoints
        #: The upper bounds of each scan point in each axis for fly-scanning
        self.upper = upper or midpoints
        if gap is not None:
            #: Whether there is a gap between this frame and the previous. First
            #: element is whether there is a gap between the last frame and the first
            self.gap = gap
        else:
            # Need to calculate gap as not passed one
            # We have a gap if upper[i] != lower[i+1] for any axes
            axes_gap = [
                np.roll(u, 1) != l
                for u, l in zip(self.upper.values(), self.lower.values())
            ]
            self.gap = np.logical_or.reduce(axes_gap)
        # Check all axes and ordering are the same
        assert list(self.midpoints) == list(self.lower) == list(self.upper), (
            f"Mismatching axes "
            f"{list(self.midpoints)} != {list(self.lower)} != {list(self.upper)}"
        )
        # Check all lengths are the same
        lengths = set(
            len(arr)
            for d in (self.midpoints, self.lower, self.upper)
            for arr in d.values()
        )
        lengths.add(len(self.gap))
        assert len(lengths) <= 1, f"Mismatching lengths {list(lengths)}"

    def axes(self) -> List[K]:
        """The axes that are present in `midpoints`, `lower` and `upper`
        which will move during the scan"""
        return list(self.midpoints.keys())

    def __len__(self) -> int:
        """The number of `stack` in the scan"""
        # All axespoints arrays are same length, pick the first one
        return len(self.gap)

    def extract(self: F, indices: np.ndarray, for_path=False) -> "Frames[K]":
        """Return a new Frames that produces this dimension
        restricted to the indices provided

        >>> dim = Frames({"x": np.array([1, 2, 3])})
        >>> dim.extract(np.array([1, 0, 1])).midpoints
        {'x': array([2, 1, 2])}
        """
        dim_indices = indices % len(self)

        def extract_dict(ds: Iterable[AxesPoints[K]]) -> AxesPoints[K]:
            for d in ds:
                return {k: v[dim_indices] for k, v in d.items()}
            return {}

        def extract_gap(gaps: Iterable[np.ndarray]) -> Optional[np.ndarray]:
            for gap in gaps:
                if for_path:
                    return gap[dim_indices]
            return None

        return _merge_frames(self, dict_merge=extract_dict, gap_merge=extract_gap)

    def concat(self: F, other: F, gap: bool = False) -> F:
        """Return a new Frames with arrays from self and other concatenated
        together. Require both Frames to have the same axes and snake
        settings

        Args:
            other: The dimension to concatenate to self
            snake: Whether to force a gap between the two dimensions

        >>> dim = Frames({"x": np.array([1, 2, 3])})
        >>> dim2 = Frames({"x": np.array([5, 6, 7])})
        >>> dim.concat(dim2).midpoints
        {'x': array([1, 2, 3, 5, 6, 7])}
        """
        assert self.axes() == other.axes(), f"axes {self.axes()} != {other.axes()}"

        def concat_dict(ds: Sequence[AxesPoints[K]]) -> AxesPoints[K]:
            # Concat each array in midpoints, lower, upper. E.g.
            # lower[ax] = np.concatenate(self.lower[ax], other.lower[ax])
            return {a: np.concatenate([d[a] for d in ds]) for a in self.axes()}

        def concat_gap(gaps: Sequence[np.ndarray]) -> np.ndarray:
            g = np.concatenate(gaps)
            # Calc the first point
            g[0] = gap_between_frames(other, self)
            # And the join point
            g[len(self)] = gap or gap_between_frames(self, other)
            return g

        return _merge_frames(self, other, dict_merge=concat_dict, gap_merge=concat_gap)

    def zip(self: F, other: F) -> F:
        """Return a new Frames with arrays from axes of self and other
        merged together. Require both Frames to not share axes, and
        to have the snake settings

        >>> dimx = Frames({"x": np.array([1, 2, 3])})
        >>> dimy = Frames({"y": np.array([5, 6, 7])})
        >>> dimx.zip(dimy).midpoints
        {'x': array([1, 2, 3]), 'y': array([5, 6, 7])}
        """
        overlapping = list(set(self.axes()).intersection(other.axes()))
        assert not overlapping, f"Zipping would overwrite axes {overlapping}"

        def zip_dict(ds: Sequence[AxesPoints[K]]) -> AxesPoints[K]:
            # Merge dicts for midpoints, lower, upper. E.g.
            # lower[ax] = {**self.lower[ax], **other.lower[ax]}
            return dict(kv for d in ds for kv in d.items())

        def zip_gap(gaps: Sequence[np.ndarray]) -> np.ndarray:
            # Gap if either dim has a gap. E.g.
            # gap[i] = self.gap[i] | other.gap[i]
            return np.logical_or.reduce(gaps)

        return _merge_frames(self, other, dict_merge=zip_dict, gap_merge=zip_gap)


def _merge_frames(
    *stack: F,
    dict_merge=Callable[[Sequence[AxesPoints[K]]], AxesPoints[K]],
    gap_merge=Callable[[Sequence[np.ndarray]], Optional[np.ndarray]],
) -> F:
    types = set(type(fs) for fs in stack)
    assert len(types) == 1, f"Mismatching types for {stack}"
    cls = types.pop()

    # If any lower or upper are different, apply to those
    kwargs = {}
    for a in ("lower", "upper"):
        if any(fs.midpoints is not getattr(fs, a) for fs in stack):
            kwargs[a] = dict_merge([getattr(fs, a) for fs in stack])

    # Apply to midpoints, force calculation of gap
    return cls(
        midpoints=dict_merge([fs.midpoints for fs in stack]),
        gap=gap_merge([fs.gap for fs in stack]),
        **kwargs,
    )


class SnakedFrames(Frames[K]):
    def __init__(
        self,
        midpoints: AxesPoints[K],
        lower: AxesPoints[K] = None,
        upper: AxesPoints[K] = None,
        gap: np.ndarray = None,
    ):
        super().__init__(midpoints, lower=lower, upper=upper, gap=gap)
        # Override first element of gap to be True, as subsequent runs
        # of snake scans are always joined end -> start
        self.gap[0] = False

    @classmethod
    def from_frames(cls: Type[F], stack: Frames[K]) -> F:
        return cls(stack.midpoints, stack.lower, stack.upper, stack.gap)

    def extract(self: F, indices: np.ndarray, for_path=False) -> Frames[K]:
        """Return a new Frames that produces this dimension
        restricted to the indices provided

        >>> dim = Frames({"x": np.array([1, 2, 3])})
        >>> dim.extract(np.array([1, 0, 1])).midpoints
        {'x': array([2, 1, 2])}
        """
        # Calculate the indices
        # E.g for len = 4
        # indices:       0123456789
        # backwards:     0000111100
        # snake_indices: 0123321001
        # gap_indices:   0123032101
        length = len(self)
        backwards = (indices // length) % 2
        snake_indices = np.where(backwards, (length - 1) - indices, indices) % length
        if for_path:
            cls = Frames
            gap = self.gap[np.where(backwards, length - indices, indices) % length]
        else:
            cls = type(self)
            gap = None

        # If lower or upper are different, apply to those
        kwargs = {}
        if self.midpoints is not self.lower:
            # If going backwards select from the opposite bound
            kwargs["lower"] = {
                k: np.where(backwards, self.upper[k][snake_indices], v[snake_indices])
                for k, v in self.lower.items()
            }
        if self.midpoints is not self.upper:
            kwargs["upper"] = {
                k: np.where(backwards, self.lower[k][snake_indices], v[snake_indices],)
                for k, v in self.upper.items()
            }

        # Apply to midpoints
        return cls(
            {k: v[snake_indices] for k, v in self.midpoints.items()}, gap=gap, **kwargs
        )


def gap_between_frames(frames1: Frames[K], frames2: Frames[K]) -> bool:
    """Return if there is a gap between last point of frames1 and first point
    of frames2"""
    return any(frames1.upper[a][-1] != frames2.lower[a][0] for a in frames1.axes())


def squash_frames(stack: List[Frames[K]], check_path_changes=True) -> Frames[K]:
    """Squash a stack of nested Frames into a single one.

    Args:
        stack: The Frames stack to squash, from slowest to fastest moving
        check_path_changes: If True then check that nesting the output
            Frames object within others will provide the same path
            as nesting the input Frames stack within others

    See Also:
        `why-squash-can-change-path`

    >>> dimx = SnakedFrames({"x": np.array([1, 2])})
    >>> dimy = Frames({"y": np.array([3, 4])})
    >>> squash_frames([dimy, dimx]).midpoints
    {'y': array([3, 3, 4, 4]), 'x': array([1, 2, 2, 1])}
    """
    path = Path(stack)
    # Comsuming a Path of these dimensions performs the squash
    # TODO: dim.tile might give better performance but is much longer
    squashed = path.consume()
    # Check that the squash is the same as the original
    if stack and isinstance(stack[0], SnakedFrames):
        squashed = SnakedFrames.from_frames(squashed)
        # The top level is snaking, so this dimension will run backwards
        # This means any non-snaking axes will run backwards, which is
        # surprising, so don't allow it
        if check_path_changes:
            non_snaking = [
                k for d in stack for k in d.axes() if not isinstance(d, SnakedFrames)
            ]
            if non_snaking:
                raise ValueError(
                    f"Cannot squash non-snaking Specs in a snaking Frames "
                    f"otherwise {non_snaking} would run backwards"
                )
    elif check_path_changes:
        # The top level is not snaking, so make sure there is an even
        # number of iterations of any snaking axis within it so it
        # doesn't jump when this dimension is iterated a second time
        for i, dim in enumerate(stack):
            # A snaking dimension within a non-snaking top level must repeat
            # an even number of times
            if isinstance(dim, SnakedFrames) and np.product(path.lengths[:i]) % 2:
                raise ValueError(
                    f"Cannot squash snaking Specs in a non-snaking Frames "
                    f"when they do not repeat an even number of times "
                    f"otherwise {dim.axes()} would jump in position"
                )
    return squashed


class Path:
    """A consumable route through one or more dimensions,
    representing a scan path.

    Args:
        stack: The Frames stack describing the scan, from slowest to fastest
            moving
        start: The index of where in the Path to start
        num: The number of scan points to produce after start. None means up to
            the end

    See Also:
        `iterate-a-spec`
    """

    def __init__(self, stack: List[Frames[K]], start: int = 0, num: int = None):
        #: The Frames stack describing the scan, from slowest to fastest moving
        self.stack = stack
        #: Index that is next to be consumed
        self.index = start
        #: The lengths of all the stack
        self.lengths = np.array([len(f) for f in stack])
        #: Index of the end point, one more than the last index that will be
        #: produced
        self.end_index = np.product(self.lengths)
        if num is not None and start + num < self.end_index:
            self.end_index = start + num

    def consume(self, num: int = None) -> Frames[K]:
        """Consume at most num points from the Path and return as a Frames

        >>> dimx = SnakedFrames({"x": np.array([1, 2])})
        >>> dimy = Frames({"y": np.array([3, 4])})
        >>> path = Path([dimy, dimx])
        >>> path.consume(3).midpoints
        {'y': array([3, 3, 4]), 'x': array([1, 2, 2])}
        >>> path.consume(3).midpoints
        {'y': array([4]), 'x': array([1])}
        >>> path.consume(3).midpoints
        {'y': array([], dtype=int64), 'x': array([], dtype=int64)}
        """
        if num is None:
            end_index = self.end_index
        else:
            end_index = min(self.index + num, self.end_index)
        indices = np.arange(self.index, end_index)
        self.index = end_index
        stack = Frames({}, {}, {}, np.zeros(indices.shape, dtype=np.bool_))
        # Example numbers below from a 2x3x4 ZxYxX scan
        for i, dim in enumerate(self.stack):
            # Number of times each point will repeat: Z:12, Y:4, X:1
            repeats = np.product(self.lengths[i + 1 :])
            # Scan indices mapped to indices within dimension:
            # Z:000000000000111111111111
            # Y:000011112222000011112222
            # X:012301230123012301230123
            if repeats > 1:
                dim_indices = indices // repeats
            else:
                dim_indices = indices
            # Create the sliced dim
            sliced = dim.extract(dim_indices, for_path=True)
            if repeats > 1:
                # Whether this dim contributes to the gap bit
                # Z:000000000000100000000000
                # Y:000010001000100010001000
                # X:111111111111111111111111
                in_gap = (indices % repeats) == 0
                # If in_gap, then keep the relevant gap bit
                sliced.gap &= in_gap
            # Zip it with the output dimension
            stack = stack.zip(sliced)
        return stack

    def __len__(self) -> int:
        """Number of points left in a scan, reduces when `consume` is called"""
        return self.end_index - self.index


class Midpoints:
    """Convenience iterable that produces the scan points for each axis. For
    better performance, consume from a `Path` instead.

    Args:
        stack: The stack of Frames describing the scan, from slowest to fastest
            moving

    See Also:
        `iterate-a-spec`

    >>> dimx = SnakedFrames({"x": np.array([1, 2])})
    >>> dimy = Frames({"y": np.array([3, 4])})
    >>> mp = Midpoints([dimy, dimx])
    >>> for p in mp: print(p)
    {'y': 3, 'x': 1}
    {'y': 3, 'x': 2}
    {'y': 4, 'x': 2}
    {'y': 4, 'x': 1}
    """

    def __init__(self, stack: List[Frames[K]]):
        #: The stack of Frames describing the scan, from slowest to fastest moving
        self.stack = stack

    @property
    def axes(self) -> List:
        """The axes that will be present in each points dictionary"""
        axes = []
        for dim in self.stack:
            axes += dim.axes()
        return axes

    def __len__(self) -> int:
        """The number of dictionaries that will be produced if iterated over"""
        return np.product([len(dim) for dim in self.stack])

    def __iter__(self) -> Iterator[AxesPoints[K]]:
        path = Path(self.stack)
        while len(path):
            dim = path.consume(1)
            yield {a: dim.midpoints[a][0] for a in dim.axes()}
