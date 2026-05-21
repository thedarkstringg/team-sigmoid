from src.services import nutrition_cache as cache


def setup_function():
    cache.clear()


def teardown_function():
    cache.clear()


def test_cache_set_get_size_and_clear():
    assert cache.size() == 0
    assert cache.get("rice") is None

    data = {"kcal": 130}
    cache.set("rice", data, ttl=60)

    assert cache.size() == 1
    assert cache.get("rice") == data

    cache.clear()
    assert cache.size() == 0
    assert cache.get("rice") is None


def test_cache_expired_entry_returns_none_and_removes_item():
    cache.set("expired rice", {"kcal": 130}, ttl=-1)

    assert cache.size() == 1
    assert cache.get("expired rice") is None
    assert cache.size() == 0


def test_cache_invalidate_removes_single_entry():
    cache.set("rice", {"kcal": 130}, ttl=60)
    cache.set("chicken", {"kcal": 165}, ttl=60)

    cache.invalidate("rice")

    assert cache.get("rice") is None
    assert cache.get("chicken") == {"kcal": 165}
    assert cache.size() == 1


def test_cache_invalidate_missing_key_is_safe():
    cache.invalidate("missing")
    assert cache.size() == 0
