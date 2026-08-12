def inf_iterator(iterable):
    """An infinite iterator

    Args:
        iterable (iterable): the iterable to iterate over

    Yields:
        any: the next element in the iterable
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def inf_iterator_with_prefix(iterable, prefix_iterable=None):
    """An infinite iterator over `iterable`, optionally yielding a `prefix_iterable` first.

    Used to resume an infinite stream mid-pass: `prefix_iterable` supplies the
    remainder of the current (possibly skip-adjusted) pass exactly once, after
    which the loop iterates `iterable` (the full, unmodified iterable) forever.

    Args:
        iterable (iterable): the iterable to loop over indefinitely after the prefix is exhausted
        prefix_iterable (iterable, optional): an iterable to fully exhaust once before looping. Defaults to None.

    Yields:
        any: the next element in the sequence
    """
    if prefix_iterable is not None:
        yield from prefix_iterable
    while True:
        yield from iterable
