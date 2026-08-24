from dataclasses import dataclass


@dataclass(frozen=True)
class FileSpec:
    prefix: str
    table: str
    source_columns: tuple[str, ...]
    db_columns: tuple[str, ...]
    min_rows: int


SPECS = (
    FileSpec(
        "FS", "federal_subject",
        ("IdSubject", "SubjectName", "lexKey1", "lexKey2"),
        ("id_subject", "subject_name", "lex_key1", "lex_key2"),
        80,
    ),
    FileSpec(
        "DT", "district",
        ("IdDistrict", "IdSubject", "DistrictName", "lexKey1", "lexKey2"),
        ("id_district", "id_subject", "district_name", "lex_key1", "lex_key2"),
        1000,
    ),
    FileSpec(
        "PC", "postal_code",
        ("PostalCode", "ThreeDigitFlag", "FederalSubjectFlag", "DistrictFlag", "MainCityFlag", "CityFlag", "StreetFlag"),
        ("postal_code", "three_digit_flag", "federal_subject_flag", "district_flag", "main_city_flag", "city_flag", "street_flag"),
        50_000,
    ),
    FileSpec(
        "MC", "main_city",
        ("IdMainCity", "IdSubject", "IdDistrict", "MainCityName", "lexKey1", "lexKey2"),
        ("id_main_city", "id_subject", "id_district", "main_city_name", "lex_key1", "lex_key2"),
        100_000,
    ),
    FileSpec(
        "CT", "city",
        ("IdCity", "IdMainCity", "CityName", "lexKey1", "lexKey2"),
        ("id_city", "id_main_city", "city_name", "lex_key1", "lex_key2"),
        10_000,
    ),
    FileSpec(
        "SR", "street",
        ("IdStreet", "StreetName", "Qualifier", "lexKey1", "lexKey2"),
        ("id_street", "street_name", "qualifier", "lex_key1", "lex_key2"),
        500_000,
    ),
    FileSpec(
        "AR", "address_range",
        (
            "IdAddress", "PostalCode", "PostOfficeName", "IdSubject", "IdDistrict", "IdMainCity",
            "IdCity", "IdStreet", "From_HouseNumber", "To_HouseNumber", "From_BuildingNumber",
            "To_Buildingnumber", "Even_odd_numbers_indiator",
        ),
        (
            "id_address", "postal_code", "post_office_name", "id_subject", "id_district", "id_main_city",
            "id_city", "id_street", "from_house_number", "to_house_number", "from_building_number",
            "to_building_number", "even_odd_indicator",
        ),
        1_000_000,
    ),
)

SPEC_BY_PREFIX = {s.prefix: s for s in SPECS}
