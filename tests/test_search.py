import numpy as np

from pipeline.demo import build_demo
from pipeline.embed import DIMS, _truncate_normalize, embed_passages
from search import coverage, fuse, keyword_search, query_language, refuse, thresholds, tokens


def test_empty_and_zero_embeddings_are_finite_without_loading_model():
    assert embed_passages([]).shape == (0, DIMS)
    assert np.isfinite(_truncate_normalize(np.zeros((1, DIMS), np.float32))).all()


def test_multilingual_keyword_search(tmp_path):
    bundle = tmp_path / "demo.sqlite"
    build_demo(bundle)
    for query, expected in (("water containers", "demo-cod-water"), ("carnet passation", "demo-cod-handover"),
                            ("inventario cuadernos", "demo-haiti-inventory"), ("دفتر التدريب", "demo-global-arabic")):
        result = keyword_search(bundle, query)
        assert not result["refused"] and result["results"][0]["doc_id"] == expected
    assert keyword_search(bundle, "quantum spacecraft")["refused"]
    assert keyword_search(bundle, "the and for")["refused"]


def test_refusal_and_fusion_contract():
    s = {"meta": {"T_vec": "0.6", "T_kw": "0.8", "T_vec_fr": "0.7"}, "language": "fr",
         "best_cos": 0.65, "best_kw": 0.5, "vec_ranked": [(1, .65), (2, .6)], "kw_ranked": [(2, .5)]}
    assert refuse(s)
    assert not refuse(s, t_vec=.6)
    assert fuse(s)[0] == 2
    assert thresholds(s["meta"], "en") == (.6, .8)
    assert tokens("Qué QUÉ cuadernos, cuadernos 2026?") == ["cuadernos"]
    assert coverage("Des étiquettes", ["etiquette"]) == 1
    assert query_language("كيف نسجل الملفات؟") == "ar"
