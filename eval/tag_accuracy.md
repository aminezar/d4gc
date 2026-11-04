# Country and year tagging

This is a held-out sample of 30 documents: 16 in English, 5 in Arabic, 5 in French and 4 in Spanish, with no more than 3 per genre in each language. None of them were used when tuning the rules. The results come from the current `pipeline/core_tags.py`. The inputs are the same as in `build.py`: `countries(extract_title, clean_text)` and `year(raw, genre, source_file)`.

In the year column, "meta" means the year comes from the HTML page's `datePublished`, which is taken as ground truth.

| Doc | Lang | Genre | Predicted countries | Judged countries | Pred. year | Judged year | Verdict |
|---|---|---|---|---|---|---|---|
| D9feb5dac85 | en | Peer-reviewed Article | Palestine, Egypt | Palestine, Egypt | – | 2025 (poster header, no marker) | country ok; year missed |
| D905421131c | en | Project Update | Syria | Syria | 2023 | 2023 (meta) | ok |
| D36fbf9932e | en | Statement | – | – (news listing page) | 2026 | 2026 (meta) | ok |
| Daa73c27a78 | en | Opinion | Haiti | – (global year-in-review) | 2023 | 2023 (meta) | **country wrong** |
| P09 | en | Activity Report Chapter | Nigeria | Nigeria | 2017 | report year, unverified | ok / year unverified |
| Dbc65e15071 | en | Project Update | Ethiopia | Ethiopia | 2026 | 2026 (meta) | ok |
| P32 | en | Evaluation Report | Sudan, DR Congo, Rwanda | thematic essay; these are its main cases | 2008 | 2008 | ok (debatable) |
| D597b901fc0 | en | Press Release | DR Congo | DR Congo | 2026 | 2026 (meta) | ok |
| Da5fedd82cc | en | Clinical Guideline Chapter | – | – | – | no date in doc | ok |
| D0fe93f344c | en | Project Update | DR Congo | DR Congo | 2024 | 2024 (meta) | ok |
| Db7dd33fa30 | en | Peer-reviewed Article | – | – (global cohort) | 2018 | 2018 | ok |
| D2e11dee055 | en | Press Release | South Sudan | South Sudan | 2026 | 2026 (meta) | ok |
| D705d4b38bf | en | Peer-reviewed Article | India | India (correspondence page) | 2019 | 2019 | ok |
| De7caf8ed52 | en | Clinical Guideline Chapter | – | – | – | no date in doc | ok |
| De04128df46 | en | Clinical Guideline Chapter | – | – | – | no date in doc | ok |
| Db44a39eff2 | en | Timeline | Russia | Russia (Chechnya) | – | no publication date in doc | ok |
| Dacd418953a | ar | Statement | DR Congo | DR Congo | 2026 | 2026 (meta) | ok |
| Daee4d5fc1f | ar | Press Release | Syria | Syria | 2025 | 2025 (meta) | ok |
| Df35d21f1d6 | ar | Voices from the Field | Panama | Panama (Darién) | 2022 | 2022 (meta) | ok |
| D25c399e2a3 | ar | Project Update | Palestine | Palestine | 2026 | 2026 (meta) | ok |
| Def236d742c | ar | Press Release | Sudan, Chad | Sudan, Chad | 2023 | 2023 (meta) | ok |
| D1b539fd0a2 | fr | Clinical Guideline Chapter | – | – | – | no date in doc | ok |
| D6f0a5ea412 | fr | Scientific News | Kenya | Kenya | – | no date in doc | ok |
| D3201fea5f5 | fr | Clinical Guideline Chapter | – | – | 2023 | 2023 ("Dernière mise à jour") | ok |
| D5075be3d1d | fr | Opinion | Haiti | – (global year-in-review) | 2023 | 2023 (meta) | **country wrong** |
| D00f74bcaf7 | fr | Activity Report Chapter | Nigeria | Nigeria, Cameroon, Chad, Niger | 2017 | report year, unverified | country incomplete |
| D37bbcc2a84 | es | Clinical Guideline Chapter | – | – | 2023 | 2023 ("Última actualización") | ok |
| D39f30a2a32 | es | Clinical Guideline Chapter | – | – | – | no date in doc | ok |
| D4ebb3e2295 | es | Opinion | Haiti | – (global year-in-review) | 2023 | 2023 (meta) | **country wrong** |
| D0c937b9312 | es | Activity Report Chapter | Cambodia | Cambodia | 2019 | report year, unverified | ok / year unverified |

## Summary

- **Countries: 27/30 correct.** The 3 errors are the same global opinion piece in en, fr and es, tagged Haiti. Two more rows are imperfect: D00f74bcaf7 is missing 3 of its 4 countries (counted as correct, since no wrong country is tagged), and P32 is debatable.
- **Year: a year for 22/30, with 0 known wrong.**
  - 14 come from HTML metadata and 5 from text markers, all verified.
  - 3 are report years for activity reports and are not verified.
  - 8 return None. 7 of those documents have no date. D9feb5dac85 has one (2025) that the rules don't pick up.
- **Text-only year rules on all 177 HTML pages that have `datePublished`:** 7 right, 1 wrong, 169 None. Without `source_file`, news genres mostly get no year.

## Known weak spots

- **Global pieces that open with one country.** When a world overview's first section is about one country, that country gets tagged. `extract_title` also returns the section heading ("Violence in Haiti") as the title, which makes it worse.
- **Multi-country documents.** Studies and regional crises often get only the top country, or none. Secondary countries need a strict floor.
- **Activity report chapters.** The year returned is the year the report covers, which can be one year before publication.
- **PDFs without a marker.** Posters and abstracts with no Published, ©, journal header or "updated" line get no year.
- **Ambiguous Arabic words** (المغرب, قطر, عمان) and author surnames in unusual layouts can still produce false matches. None were seen in this sample.
