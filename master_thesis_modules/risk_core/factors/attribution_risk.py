"""Simple MVP attribution risks for patient and age labels."""

from master_thesis_modules.risk_core.factors.common import clip01


def patient_attribute_risk(label: str, confidence: float = 1.0) -> float:
    if label == "yes":
        return clip01(confidence)
    if label == "no":
        return 0.0
    return 0.5 * clip01(confidence)


def age_attribute_risk(label: str, confidence: float = 1.0) -> float:
    if label == "old":
        return clip01(confidence)
    if label == "middle":
        return 0.5 * clip01(confidence)
    if label == "young":
        return 0.2 * clip01(confidence)
    return 0.5 * clip01(confidence)

