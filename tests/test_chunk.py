from pipeline.chunk import MAX_WORDS, MIN_WORDS, chunk_text


def words(n, w="word"):
    return " ".join([w] * n)


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.\n\nSecond paragraph.") == ["Hello world.\n\nSecond paragraph."]


def test_paragraphs_are_packed_and_bounded():
    text = "\n\n".join(words(60) + "." for _ in range(20))
    chunks = chunk_text(text)
    assert all(len(c.split()) <= MAX_WORDS for c in chunks)
    assert all(len(c.split()) >= MIN_WORDS for c in chunks)
    assert sum(len(c.split()) for c in chunks) == 1200


def test_long_paragraph_split_on_sentences():
    para = " ".join(words(40) + "." for _ in range(20))
    chunks = chunk_text(para)
    assert len(chunks) > 1
    assert all(len(c.split()) <= MAX_WORDS for c in chunks)
    assert all(c.endswith(".") for c in chunks)


def test_run_on_sentence_without_punctuation():
    chunks = chunk_text(words(1000))
    assert all(len(c.split()) <= MAX_WORDS + MIN_WORDS for c in chunks)
    assert sum(len(c.split()) for c in chunks) == 1000


def test_arabic_question_mark_is_a_sentence_end():
    para = " ".join(words(100, "كلمة") + "؟" for _ in range(5))
    assert len(chunk_text(para)) > 1


def test_empty():
    assert chunk_text("  \n\n ") == []
