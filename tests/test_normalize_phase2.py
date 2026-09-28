import os
from dataclasses import replace

import pandas as pd
import pytest

from src.entity_resolution.config import NormalizationConfig
from src.entity_resolution.normalize import _strip_legal_tail, normalize_address, normalize_name, normalize_record, tokenize

OFF = NormalizationConfig()
ALL_ON = NormalizationConfig(keep_digits=True, country_abbrev=True, stopwords=True, landmarks=True,
                             legal_bag=True, extract_fields=True)
TRAIN_S2 = "data_set/student_resource/dataset/train/train_source2.tsv"

TRICKY = [
    ("Sai Tech Limited Private", "Plot No-C/21, 1St Floor Sahid Nagar, Bhubaneswar, Khordha, Orissa", "India"),
    ("ÉTABLISSEMENTS RETRAITE SARL", "73 Q. DE PALUDATE, BORDEAUX", "France"),
    ("हरि सिस्टम्स प्राइवेट लिमिटेड", "PUNE, PUNE REGION, Maharashtra, H.NO 18/2", "India"),
    ("Joe's Diner, Inc.", "12 Main St., Ste 5, Hartford, CT 06103", "US"),
    ("", "", "US"),
    ("  ", "   ,  , ", "India"),
]


def baseline(name, addr):
    return tokenize(normalize_name(name)), tokenize(normalize_address(addr))


@pytest.mark.parametrize("name,addr,country", TRICKY)
def test_flags_off_reproduces_baseline(name, addr, country):
    out = normalize_record(name, addr, country, OFF)
    assert (out["name_tokens"], out["addr_tokens"]) == baseline(name, addr)
    assert set(out) == {"name_tokens", "addr_tokens"}


@pytest.mark.skipif(not os.path.exists(TRAIN_S2), reason="dataset not present")
def test_flags_off_reproduces_baseline_on_real_rows():
    df = pd.read_csv(TRAIN_S2, sep="\t", dtype=str, keep_default_na=False, nrows=5000)
    for name, addr, country in zip(df.business_name, df.business_address, df.country):
        out = normalize_record(name, addr, country, OFF)
        assert (out["name_tokens"], out["addr_tokens"]) == baseline(name, addr)


def test_country_abbrev_keeps_us_and_india_tokens_unchanged():
    cfg = replace(OFF, country_abbrev=True)
    for name, addr, country in TRICKY:
        if country in ("US", "India"):
            assert normalize_record(name, addr, country, cfg)["addr_tokens"] == baseline(name, addr)[1]


def test_keep_digits_and_letters_next_to_digits():
    cfg = replace(OFF, keep_digits=True)
    assert normalize_record("x", "2 rue alfred de vigny, Calais", "France", cfg)["addr_tokens"][0] == "2"
    assert normalize_record("x", "Block A 12, Delhi", "India", cfg)["addr_tokens"] == ["block", "a", "12", "delhi"]
    assert "a" not in normalize_record("x", "a quick road", "India", cfg)["addr_tokens"]


def test_legal_bag_strips_any_order_in_last_three_tokens():
    cfg = replace(OFF, legal_bag=True)
    assert normalize_record("Sai Tech Limited Private", "", "India", cfg)["name_tokens"] == ["sai", "tech"]
    assert normalize_record("Om Limited Private Trading", "", "India", cfg)["name_tokens"] == ["om", "trading"]
    assert normalize_record("Établissements SARL Retraite", "", "France", cfg)["name_tokens"] == ["etablissements", "retraite"]
    assert _strip_legal_tail(["private", "limited"]) == ["private", "limited"]


FR = replace(OFF, keep_digits=True, country_abbrev=True)


def fr_addr(addr, country="France"):
    return normalize_record("x", addr, country, FR)["addr_tokens"]


def test_french_street_position_rules():
    assert fr_addr("143 R DU CAINE, ROUBAIX") == ["143", "rue", "du", "caine", "roubaix"]
    assert fr_addr("5 B R. DU BELVEDERE, LEGE") == ["5bis", "rue", "du", "belvedere", "lege"]
    assert fr_addr("73 - Q. De Paludate, Bordeaux") == ["73", "quai", "de", "paludate", "bordeaux"]
    assert fr_addr("Bordeaux, R. Sainte Catherine") == ["bordeaux", "rue", "sainte", "catherine"]
    # a lone r / q elsewhere is not a street type
    assert "rue" not in fr_addr("Chez Martin R Dupont, Lyon")


def test_french_table_bis_and_cedex():
    assert fr_addr("59 bis Bd Jules Simon, 33000 Bordeaux Cedex") == [
        "59bis", "boulevard", "jules", "simon", "33000", "bordeaux"]
    assert fr_addr("12 Chem. des Vignes, St Emilion") == ["12", "chemin", "des", "vignes", "saint", "emilion"]
    assert fr_addr("3 All. des Pins") == ["3", "allee", "des", "pins"]
    assert fr_addr("15 B AVENUE PHENIX") == fr_addr("15 bis Avenue Phenix") == ["15bis", "avenue", "phenix"]
    assert fr_addr("5 B R. DU BELVEDERE") == ["5bis", "rue", "du", "belvedere"]
    assert fr_addr("N° 19 BD JULES SIMON") == fr_addr("NO 19 BD JULES SIMON") == ["19", "boulevard", "jules", "simon"]


def test_unseen_country_gets_generic_table_only():
    assert fr_addr("12 Main St, Rd 5", country="Germany") == ["12", "main", "st", "road", "5"]
    assert fr_addr("12 Main St", country="US") == ["12", "main", "street"]


def test_stopwords_per_country():
    cfg = replace(OFF, stopwords=True)
    assert normalize_record("Centre Hospitalier de la Arts", "", "France", cfg)["name_tokens"] == ["centre", "hospitalier", "arts"]
    assert normalize_record("The Bank of India", "", "India", cfg)["name_tokens"] == ["bank", "india"]
    assert normalize_record("De Souza Traders", "", "India", cfg)["name_tokens"] == ["de", "souza", "traders"]


def test_landmark_phrase_moves_out_of_address_tokens():
    cfg = replace(OFF, landmarks=True)
    out = normalize_record("x", "Shop 15, Goregaon East Near Dindoshi Bus Depot, Mumbai", "India", cfg)
    assert out["addr_tokens"] == ["shop", "15", "goregaon", "east", "mumbai"]
    assert out["addr_landmark"] == ["dindoshi", "bus", "depot"]


def test_extract_fields_postal_house_numbers_and_name_core():
    out = normalize_record("Bordeaux Maison SA", "59 Boulevard Jules Simon, 33000 Bordeaux", "France", ALL_ON)
    assert out["postal"] == "33000" and out["house_nums"] == ["59"]
    assert out["name_core"] == ["maison"]
    us = normalize_record("x", "12 Main St, Hartford, CT 06103-1234", "US", ALL_ON)
    assert us["postal"] == "06103" and us["postal_all"] == ["06103"] and us["house_nums"] == ["12"]
