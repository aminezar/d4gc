from pipeline.langid import detect

FR = "La tuberculose pulmonaire est une infection due à une bactérie qui se transmet par les gouttelettes. " * 10
EN = "The patients were treated in the clinic and the results are shown for each of the groups in this study. " * 10
AR = "يعاني الأطفال من سوء التغذية الحاد في المخيمات وتعمل الفرق الطبية على تقديم العلاج " * 10


def test_corrects_mislabelled_french():
    assert detect(FR, "en") == "fr"


def test_keeps_correct_label():
    assert detect(EN, "en") == "en"
    assert detect(AR, "ar") == "ar"


def test_arabic_detected():
    assert detect(AR, "fr") == "ar"


def test_short_or_mixed_text_keeps_fallback():
    assert detect("Annexes", "es") == "es"
    assert detect(EN[:300] + AR[:300], "fr") == "fr"
