from utilz.datastructures import List


def test_list():

    # Test basic sub-classing
    x = List()
    assert isinstance(x, list)
    assert len(x) == 0
    assert hasattr(x, "append")
    x.append(1)

    # len method
    assert len(x) == x.len()
    x.extend([2, 3, 4])
    assert x.len() == 4

    # map method
    assert x.map(lambda x: x * 2) == list(map(lambda i: i * 2, x))

    # filter method
    assert x.filter(lambda x: x > 1) == list(filter(lambda i: i > 1, x))

    # pmap; need to fix pmap first
    # x = List(range(10000))
    # x.pmap(lambda x: x * 2, n_jobs=2)
