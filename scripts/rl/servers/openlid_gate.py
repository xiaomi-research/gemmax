"""OpenLID-v3 language gate used by the translation reward server."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import regex


OPENLID_LANG_TO_CODE: dict[str, str] = {
    "Arabic": "ara_Arab",
    "Azerbaijani": "azj_Latn",
    "Bengali": "ben_Beng",
    "Bulgarian": "bul_Cyrl",
    "Burmese": "mya_Mymr",
    "Cantonese": "yue_Hant",
    "Catalan": "cat_Latn",
    "Chinese (Simplified)": "cmn_Hans",
    "Chinese (Traditional)": "cmn_Hant",
    "Croatian": "hrv_Latn",
    "Czech": "ces_Latn",
    "Danish": "dan_Latn",
    "Dutch": "nld_Latn",
    "English": "eng_Latn",
    "Finnish": "fin_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Greek": "ell_Grek",
    "Hebrew": "heb_Hebr",
    "Hindi": "hin_Deva",
    "Hungarian": "hun_Latn",
    "Indonesian": "ind_Latn",
    "Italian": "ita_Latn",
    "Japanese": "jpn_Jpan",
    "Kazakh": "kaz_Cyrl",
    "Khmer": "khm_Khmr",
    "Korean": "kor_Hang",
    "Lao": "lao_Laoo",
    "Malay": "zsm_Latn",
    "Norwegian": "nob_Latn",
    "Persian": "fas_Arab",
    "Polish": "pol_Latn",
    "Portuguese": "por_Latn",
    "Romanian": "ron_Latn",
    "Russian": "rus_Cyrl",
    "Slovak": "slk_Latn",
    "Slovenian": "slv_Latn",
    "Spanish": "spa_Latn",
    "Swedish": "swe_Latn",
    "Tagalog": "fil_Latn",
    "Tamil": "tam_Taml",
    "Thai": "tha_Thai",
    "Turkish": "tur_Latn",
    "Urdu": "urd_Arab",
    "Uzbek": "uzn_Latn",
    "Vietnamese": "vie_Latn",
}

_NONWORD_PATTERN = regex.compile(r"[^\p{Word}\p{Zs}]|\d")
_SPACE_PATTERN = regex.compile(r"\s\s+")


def target_language_to_openlid_code(target_language: str | None) -> str | None:
    """Map a MiLMMT target-language name to an OpenLID language/script code."""
    if not target_language:
        return None
    name = target_language.strip()
    if name in OPENLID_LANG_TO_CODE:
        return OPENLID_LANG_TO_CODE[name]
    base = re.sub(r"\s*\(.*?\)\s*", "", name).strip()
    if base in OPENLID_LANG_TO_CODE:
        return OPENLID_LANG_TO_CODE[base]
    head = base.split()[0] if base else ""
    return OPENLID_LANG_TO_CODE.get(head)


def preprocess_openlid(text: str) -> str:
    """Apply the preprocessing used by OpenLID-v3 before prediction."""
    normalized = (text or "").strip().replace("\n", " ").lower()
    normalized = _SPACE_PATTERN.sub(" ", normalized)
    return _NONWORD_PATTERN.sub("", normalized)


class OpenLIDGate:
    """Batch OpenLID predictor with exact language/script matching."""

    def __init__(self, model_path: str | Path, model: Any | None = None):
        if model is None:
            import fasttext

            model_path = Path(model_path)
            if not model_path.is_file():
                raise FileNotFoundError(f"OpenLID model not found: {model_path}")
            model = fasttext.load_model(str(model_path))
        self.model = model

    @staticmethod
    def _label_code(label: str) -> str:
        return label.replace("__label__", "")

    def evaluate_batch(
        self,
        translations: list[str],
        target_languages: list[str | None],
    ) -> list[dict[str, Any]]:
        if len(translations) != len(target_languages):
            raise ValueError(
                "translations and target_languages must have the same length: "
                f"{len(translations)} != {len(target_languages)}"
            )
        if not translations:
            return []

        target_codes = [target_language_to_openlid_code(language) for language in target_languages]
        cleaned = [preprocess_openlid(text) for text in translations]
        safe_inputs = [text if text else " " for text in cleaned]
        labels_batch, _ = self.model.predict(safe_inputs, k=1)

        results: list[dict[str, Any]] = []
        for index, target_code in enumerate(target_codes):
            if target_code is None:
                results.append(
                    {
                        "la_ok": 1,
                        "la_skip": 1,
                        "pred_iso": "",
                        "tgt_iso": "",
                    }
                )
                continue

            labels = labels_batch[index]
            predicted_code = self._label_code(labels[0]) if labels and cleaned[index] else ""
            results.append(
                {
                    "la_ok": int(predicted_code == target_code),
                    "la_skip": 0,
                    "pred_iso": predicted_code,
                    "tgt_iso": target_code,
                }
            )
        return results
