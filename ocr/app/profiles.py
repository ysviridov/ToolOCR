from __future__ import annotations

from dataclasses import dataclass

from .gost_r_51506_99 import (
    DOMESTIC_NO_WINDOW_FIGURES,
    DOMESTIC_WINDOW_FIGURES,
    ENVELOPE_SPECS,
    DomesticLayout,
    EnvelopeFormat,
)


@dataclass(frozen=True, slots=True)
class ShipmentProfile:
    """Профиль оформления внутреннего почтового отправления по ГОСТ Р 51506-99."""

    profile_id: str
    format: EnvelopeFormat
    layout: DomesticLayout
    window: bool
    figure: str
    width_mm: float
    height_mm: float


# На первом инкременте Stage 2.1 распознаём внутренние отправления (Вн).
# Для конвертов без окна используются рисунки приложения А, для конвертов
# с окном — приложения Б. Окно допускается только для C6, DL и C5.

def _build_domestic_profiles() -> tuple[ShipmentProfile, ...]:
    profiles: list[ShipmentProfile] = []

    for envelope_format, spec in ENVELOPE_SPECS.items():
        for layout in DomesticLayout:
            profiles.append(
                ShipmentProfile(
                    profile_id=f"vn-{envelope_format.value.lower()}-{layout.value.lower()}",
                    format=envelope_format,
                    layout=layout,
                    window=False,
                    figure=DOMESTIC_NO_WINDOW_FIGURES[layout],
                    width_mm=spec.width_mm,
                    height_mm=spec.height_mm,
                )
            )

            if spec.window_allowed:
                profiles.append(
                    ShipmentProfile(
                        profile_id=f"vn-{envelope_format.value.lower()}o-{layout.value.lower()}",
                        format=envelope_format,
                        layout=layout,
                        window=True,
                        figure=DOMESTIC_WINDOW_FIGURES[layout],
                        width_mm=spec.width_mm,
                        height_mm=spec.height_mm,
                    )
                )

    return tuple(profiles)


DOMESTIC_PROFILES = _build_domestic_profiles()
PROFILE_BY_ID = {profile.profile_id: profile for profile in DOMESTIC_PROFILES}


def profiles_for_format(envelope_format: EnvelopeFormat) -> tuple[ShipmentProfile, ...]:
    return tuple(profile for profile in DOMESTIC_PROFILES if profile.format == envelope_format)
