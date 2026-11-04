import json
import re

import pytest

from pipeline import core_tags as ct


def tags(text: str, title: str = "") -> list[str]:
    return ct.countries(title, text)


# --------------------------------------------------------------------------
# countries.json
# --------------------------------------------------------------------------


def test_gazetteer_shape():
    data = json.loads(ct.COUNTRIES_JSON.read_text(encoding="utf-8"))
    isos = [c["iso3"] for c in data]
    assert len(isos) == len(set(isos)) >= 199  # 193 UN members + 6 others
    assert all(re.fullmatch(r"[A-Z]{3}", i) for i in isos)
    names = {c["name"] for c in data}
    for n in ("DR Congo", "South Sudan", "Syria", "Palestine", "Kosovo",
              "Western Sahara", "Somaliland", "Taiwan", "Venezuela"):
        assert n in names


def test_strong_aliases_are_unambiguous():
    # Only weak aliases ("Congo", "Korea") may point at several countries.
    _, table, _, _ = ct._gazetteer()
    for alias, entries in table.items():
        if len({e[0] for e in entries}) > 1:
            assert all(e[1] for e in entries), alias


@pytest.mark.parametrize("name,code", [
    ("DR Congo", "COD"), ("Congo", "COG"), ("South Sudan", "SSD"), ("Sudan", "SDN"),
    ("Niger", "NER"), ("Nigeria", "NGA"), ("Guinea", "GIN"), ("Guinea-Bissau", "GNB"),
    ("Equatorial Guinea", "GNQ"), ("Papua New Guinea", "PNG"), ("Palestine", "PSE"),
    ("Kosovo", "XKX"), ("Venezuela", "VEN"), ("Côte d'Ivoire", "CIV"),
])
def test_iso3(name, code):
    assert ct.iso3(name) == code


# --------------------------------------------------------------------------
# Ambiguous names
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("MSF runs a cholera treatment centre in Guinea-Bissau.", ["Guinea-Bissau"]),
    ("MSF runs a cholera treatment centre in Equatorial Guinea.", ["Equatorial Guinea"]),
    ("MSF runs a malaria project in Papua New Guinea.", ["Papua New Guinea"]),
    ("MSF runs an Ebola treatment centre in Guinea.", ["Guinea"]),
    ("Les équipes de MSF interviennent en Guinée-Bissau.", ["Guinea-Bissau"]),
    ("The vaccine was first tested on guinea pigs in the laboratory.", []),
    ("Malnutrition is rising in Niger, in the Maradi region.", ["Niger"]),
    ("Malnutrition is rising in northern Nigeria, in Katsina state.", ["Nigeria"]),
    ("Oil spills in the Niger Delta have harmed fishing communities.", ["Nigeria"]),
    ("Aspergillus niger was isolated from the samples.", []),
    ("Fighting continues in South Sudan, around Malakal.", ["South Sudan"]),
    ("Fighting continues in Sudan, around Khartoum.", ["Sudan"]),
    ("Le choléra se propage au Soudan du Sud.", ["South Sudan"]),
    ("Samples were analysed at the CDC in Atlanta, Georgia.", []),
    ("MSF treats TB patients in Georgia and Abkhazia.", ["Georgia"]),
    ("Syrian refugees in Jordan face rising costs.", ["Jordan"]),
    ("The study was led by Michael Jordan and colleagues.", []),
    ("We roasted a turkey for the staff party.", []),
    ("Refugees from Sudan arrive daily in eastern Chad. " * 3, ["Chad", "Sudan"]),
])
def test_ambiguous_names(text, expected):
    assert sorted(tags(text)) == sorted(expected)


def test_congo_variants():
    assert tags("Ebola outbreak in North Kivu, Democratic Republic of the Congo.") == ["DR Congo"]
    assert tags("Épidémie de rougeole en RDC, dans la province de l'Ituri.") == ["DR Congo"]
    assert tags("Measles in the Republic of the Congo, near Brazzaville.") == ["Congo"]
    # bare "Congo"/"Congolese" go to the Congo with unambiguous evidence
    text = "MSF teams in Goma, DRC, treated Congolese patients. Eastern Congo remains unstable."
    assert tags(text) == ["DR Congo"]
    # and are dropped when there is none
    assert tags("Congolese music filled the room.") == []


def test_acronyms_are_case_sensitive():
    assert tags("Violence in the CAR has displaced thousands; MSF works in Bangui, CAR.") == [
        "Central African Republic"]
    assert tags("The car broke down on the way to the clinic.") == []


def test_arabic_prefixes_and_ambiguity():
    assert tags("تعمل فرقنا في جنوب السودان منذ سنوات") == ["South Sudan"]
    assert tags("وصل اللاجئون إلى تشاد قادمين من السودان والسودان يشهد حربا. " * 3) == ["Sudan", "Chad"]
    assert tags("تقديم الرعاية للسودان وبالسودان") == ["Sudan"]  # ل + ال contracts to لل
    assert tags("الوضع في جمهورية الكونغو الديمقراطية يتدهور") == ["DR Congo"]
    # مالي also means "financial"
    assert tags("نحتاج إلى الدعم المالي والدعم مالي للمشروع") == []


def test_affiliations_do_not_count():
    text = ("Jane Doe1, John Roe2\n1 Médecins Sans Frontières, Brussels, Belgium\n"
            "2 London School of Hygiene, London, UK\n"
            "Abstract: We describe a cholera outbreak in Haiti. Cases in Haiti peaked in "
            "October. Data were analysed with Stata (StataCorp, College Station, TX, USA).")
    assert tags(text) == ["Haiti"]
    assert tags("MSF-France and MSF Spain sent teams to Yemen; Yemen's hospitals are full.") == ["Yemen"]


def test_list_of_countries_is_not_an_address():
    text = "MSF responded to cholera outbreaks in Malawi, Mozambique, Zambia. " * 3
    assert set(tags(text)) == {"Malawi", "Mozambique", "Zambia"}


def test_global_document_has_no_country():
    text = ("Amoxicillin is the first-line treatment for community-acquired pneumonia in "
            "children. Give 80 to 100 mg/kg daily in two divided doses for 5 days. ") * 20
    assert tags(text, "Pneumonia in children") == []


def test_title_country_is_kept_and_first():
    text = "This is a chapter about cholera. " * 30 + "A brief note on Kenya." + " Cholera spreads." * 30
    assert tags(text, "Cholera response in Haiti")[0] == "Haiti"


def test_reference_list_is_ignored():
    body = "We studied malaria in Mali. Mali has high transmission. " * 5
    refs = "\nReferences\n" + "".join(f"{i}. Smith J. Malaria in Kenya. Lancet 2010.\n" for i in range(20))
    assert tags(body + "Discussion follows. " * 40 + refs) == ["Mali"]


def test_palestine_documents_do_not_tag_israel():
    text = ("Israeli forces launched incursions in the West Bank, Palestine. Israel must allow "
            "aid into Gaza. MSF teams in Gaza and Jenin report attacks. ") * 4
    assert tags(text) == ["Palestine"]


def test_many_country_overview_is_global():
    countries = ["Haiti", "Yemen", "Syria", "Sudan", "South Sudan", "Nigeria", "Afghanistan", "Ukraine"]
    text = " ".join(f"In {c}, MSF teams provided care. {c} remains in crisis." for c in countries) * 3
    assert tags(text) == []


# --------------------------------------------------------------------------
# Year
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raw,genre,expected", [
    ("Correspondence\nwww.thelancet.com   Vol 381   February 23, 2013\n625\nThe lack of ...",
     "Peer-reviewed Article", 2013),
    ("SYSTEMATIC REVIEW\npublished: 20 August 2020\ndoi: 10.3389/fpubh", "Peer-reviewed Article", 2020),
    ("Alba S, et al. BMJ Global Health 2020;5:e003236. doi:10.1136/bmjgh-2020-003236",
     "Peer-reviewed Article", 2020),
    ("Tornimbene et al. BMC Proceedings (2025) 19:7 https://doi.org/10.1186/s12919",
     "Peer-reviewed Article", 2025),
    ("Accepted Manuscript\n© The Author(s) 2019. Published by Oxford University Press.",
     "Peer-reviewed Article", 2019),
    ("Received October 15, 2013; Accepted February 13, 2014; Published March 20, 2014",
     "Peer-reviewed Article", 2014),
    ("Sommaire Mise à jour : Janvier 2022 16.1 Introduction", "Clinical Guideline Chapter", 2022),
    ("Última actualización: Octubre 2023 ACT | combinación", "Clinical Guideline Chapter", 2023),
    ("Last Update - June 2026 Select language", "Clinical Guideline Chapter", 2026),
    ("Soins obstétricaux et néonatals essentiels. Édition 2019", "Clinical Guideline Chapter", 2019),
    ("تاريخ النشر: ٢٨ أكتوبر/تشرين الأول ٢٠٢٤", "Press Release", 2024),
    ("MSF: Open statement on the launch of the Hub\n22 May 2018\nMédecins Sans Frontières welcomes",
     "Statement", 2018),
    ("**International Activity Report 2019** > South Sudan. In 2019, MSF ran 19 projects.",
     "Activity Report Chapter", 2019),
])
def test_year_formats(raw, genre, expected):
    assert ct.year(raw, genre) == expected


@pytest.mark.parametrize("raw,genre", [
    ("Published online 1 September 1965", "Peer-reviewed Article"),        # before MSF existed
    ("Published online 1 September 2031", "Peer-reviewed Article"),        # in the future
    ("In November 2025, the government announced new restrictions.", "Project Update"),  # event
    ("MSF has worked here since May 2021 and saw 5,000 patients.", "Press Release"),     # bound
    ("# Zambia\nRatified conventions - 1966 - 1st Geneva Convention", "Legal Encyclopedia Entry"),
    ("Amoxicillin is first-line. Give 80 mg/kg/day.", "Clinical Guideline Chapter"),
    ("", "Press Release"),
])
def test_year_none_when_unsure(raw, genre):
    assert ct.year(raw, genre) is None


def test_year_ignores_reference_list_citations():
    raw = ("Understanding displacement: a qualitative study\n" + "Body text. " * 300
           + "\n4. Smith J. Poets after Nakba. Int J Polit Cult Soc 2008; 22: 37-56.\n")
    assert ct.year(raw, "Peer-reviewed Article") is None


def test_year_prefers_html_metadata(tmp_path):
    page = tmp_path / "doc.html"
    page.write_text('<script type="application/ld+json">{"datePublished": "2025-10-27T16:44:20+00:00"}'
                    '</script><p>While MSF left El Fasher in August 2024, ...</p>', encoding="utf-8")
    assert ct.html_year(page) == 2025
    assert ct.year("While MSF left El Fasher in August 2024, ...", "Statement", page) == 2025
    # falls back to the text when there is no metadata
    bare = tmp_path / "bare.html"
    bare.write_text("<p>nothing</p>", encoding="utf-8")
    assert ct.year("Published: 3 March 2021", "Peer-reviewed Article", bare) == 2021
    assert ct.html_year(tmp_path / "missing.pdf") is None


# --------------------------------------------------------------------------
# DR Congo regressions (demo bundle)
# --------------------------------------------------------------------------

FILLER = "Training notebooks were counted at the practice desk and totals were recorded. "


def test_long_article_naming_drc_a_dozen_times():
    # Long trial report: DR Congo only in the body, log-damped; must still pass.
    head = ("Improving Ebola virus disease outbreak control\nJ Doe1\n1 University of Kinshasa, "
            "Kinshasa, Democratic Republic of the Congo\nAbstract\n" + FILLER * 20)
    body = "".join(FILLER * 40 + "During the tenth outbreak in the Democratic Republic of the Congo, "
                   "teams vaccinated contacts. " for _ in range(11))
    body += "Earlier outbreaks in Guinea were similar. " * 2
    assert tags(head + body)[:1] == ["DR Congo"]


def test_bare_congo_counts_for_drc_with_a_drc_signal():
    text = (FILLER * 10 + "Patients were enrolled in the Republic of the Congo and the Democratic "
            "Republic of the Congo (DR Congo). " + FILLER * 30
            + "The trial in Congo enrolled 280 patients. " * 10)
    assert tags(text)[0] == "DR Congo"


def test_bare_congo_without_any_signal_is_drc():
    assert tags("MSF ran a sleeping sickness trial in Congo. " * 5) == ["DR Congo"]


def test_republic_of_congo_documents_still_work():
    text = ("Measles outbreak in the Republic of the Congo: MSF teams in Brazzaville and in "
            "Pointe-Noire vaccinated children. Congo has seen several outbreaks. ") * 3
    assert tags(text) == ["Congo"]


def test_world_overview_report_has_no_country():
    countries = ["DR Congo", "Yemen", "Syria", "Sudan", "South Sudan", "Nigeria", "Afghanistan",
                 "Haiti", "Somalia", "Ethiopia", "Iraq", "Lebanon", "Niger", "Chad", "Mali",
                 "Kenya", "Uganda", "Malawi"]
    text = "MSF International Financial Report. Geneva, Switzerland. " + "".join(
        f"{c}: expenditure 12.3 million; programmes in {c} ran all year. " * 4 for c in countries)
    assert tags(text, "Financial report 2019") == []


@pytest.mark.parametrize("doc_id", ["Dd0941288db", "De3f857b161"])
def test_corpus_drc_documents(doc_id):
    from pipeline.load import ROOT
    path = ROOT / "data" / "parsed" / f"{doc_id}.md"
    if not path.exists():
        pytest.skip("corpus not available")
    from pipeline import clean
    raw = path.read_text(encoding="utf-8")
    text = clean.clean_text(raw, "pdf")
    assert ct.countries(clean.extract_title(raw, "pdf", text), text)[0] == "DR Congo"
