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
