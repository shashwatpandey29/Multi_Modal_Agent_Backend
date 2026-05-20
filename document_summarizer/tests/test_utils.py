from document_summarizer import cache


def test_hash_and_key():
    h1 = cache.hash_text("Hello World")
    h2 = cache.hash_text("hello world")
    assert isinstance(h1, str) and len(h1) == 64
    assert h1 != h2  # case-sensitive hash

    k = cache.make_key("search", 123, "abc")
    assert k == "search:123:abc"
