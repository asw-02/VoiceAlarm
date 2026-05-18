#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small text formatting helpers for assistant replies and Piper input.
"""

import re


NUMBER_WORDS = {
    0: "null",
    1: "eins",
    2: "zwei",
    3: "drei",
    4: "vier",
    5: "f\u00fcnf",
    6: "sechs",
    7: "sieben",
    8: "acht",
    9: "neun",
    10: "zehn",
    11: "elf",
    12: "zw\u00f6lf",
    13: "dreizehn",
    14: "vierzehn",
    15: "f\u00fcnfzehn",
    16: "sechzehn",
    17: "siebzehn",
    18: "achtzehn",
    19: "neunzehn",
    20: "zwanzig",
    30: "drei\u00dfig",
    40: "vierzig",
    50: "f\u00fcnfzig",
}

ORDINAL_WORDS = {
    1: "ersten",
    2: "zweiten",
    3: "dritten",
    4: "vierten",
    5: "f\u00fcnften",
    6: "sechsten",
    7: "siebten",
    8: "achten",
    9: "neunten",
    10: "zehnten",
    11: "elften",
    12: "zw\u00f6lften",
    13: "dreizehnten",
    14: "vierzehnten",
    15: "f\u00fcnfzehnten",
    16: "sechzehnten",
    17: "siebzehnten",
    18: "achtzehnten",
    19: "neunzehnten",
    20: "zwanzigsten",
    30: "drei\u00dfigsten",
}

MONTH_WORDS = {
    1: "Januar",
    2: "Februar",
    3: "M\u00e4rz",
    4: "April",
    5: "Mai",
    6: "Juni",
    7: "Juli",
    8: "August",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Dezember",
}


def number_to_german(number):
    if number in NUMBER_WORDS:
        return NUMBER_WORDS[number]

    if 21 <= number <= 59:
        ones = number % 10
        tens = number - ones
        ones_word = "ein" if ones == 1 else NUMBER_WORDS[ones]
        return f"{ones_word}und{NUMBER_WORDS[tens]}"

    return str(number)


def _spoken_clock_hour(hour):
    hour = hour % 12
    if hour == 0:
        hour = 12
    return number_to_german(hour)


def ordinal_to_german(number):
    if number in ORDINAL_WORDS:
        return ORDINAL_WORDS[number]

    if 21 <= number <= 31:
        return f"{number_to_german(number)}sten"

    return str(number)


def time_to_german_words(hour, minute):
    if minute == 15:
        return f"viertel nach {_spoken_clock_hour(hour)}"

    if minute == 30:
        return f"halb {_spoken_clock_hour(hour + 1)}"

    if minute == 45:
        return f"viertel vor {_spoken_clock_hour(hour + 1)}"

    hour_word = "ein" if hour == 1 else number_to_german(hour)

    if minute == 0:
        return f"{hour_word} Uhr"

    return f"{hour_word} Uhr {number_to_german(minute)}"


def date_to_german_words(value):
    """Format a date for natural German speech."""

    year = value.year
    if 2001 <= year <= 2099:
        year_word = number_to_german(year % 100)
    else:
        year_word = str(year)

    return f"den {ordinal_to_german(value.day)} {MONTH_WORDS[value.month]} {year_word}"


def format_for_tts(text):
    """Convert numeric clock times into a form Piper reads naturally."""

    def replace_time(match):
        hour = int(match.group(1))
        minute = int(match.group(2))
        return time_to_german_words(hour, minute)

    return re.sub(
        r"\b([01]?\d|2[0-3]):([0-5]\d)\s*Uhr\b",
        replace_time,
        text,
        flags=re.IGNORECASE,
    )


def make_reply_informal(text):
    """Patch common formal fallback phrases into the assistant's Du style."""

    replacements = [
        (
            "Ich verstehe nicht\\.\\s*K(?:oe|\u00f6)nnen Sie (?:es |das )?bitte wiederholen\\??",
            "Ich verstehe nicht. Kannst du es bitte wiederholen?",
        ),
        (
            "K(?:oe|\u00f6)nnen Sie (?:es |das )?bitte wiederholen\\?",
            "Kannst du es bitte wiederholen?",
        ),
        (
            "K(?:oe|\u00f6)nnten Sie (?:es |das )?bitte wiederholen\\?",
            "Kannst du es bitte wiederholen?",
        ),
        (
            "Wiederholen Sie (?:es |das )?bitte\\.",
            "Kannst du es bitte wiederholen?",
        ),
        ("K(?:oe|\u00f6)nnen Sie\\b", "Kannst du"),
        ("K(?:oe|\u00f6)nnten Sie\\b", "Kannst du"),
        ("M(?:oe|\u00f6)chten Sie\\b", "Moechtest du"),
        ("Haben Sie\\b", "Hast du"),
        ("Sind Sie\\b", "Bist du"),
        ("Wollen Sie\\b", "Willst du"),
        ("Brauchen Sie\\b", "Brauchst du"),
        ("Sagen Sie\\b", "Sag"),
        ("Ich kann Ihnen\\b", "Ich kann dir"),
        ("f(?:ue|\u00fc)r Sie\\b", "fuer dich"),
        ("bei Ihnen\\b", "bei dir"),
        ("Ihnen\\b", "dir"),
        ("Ihre\\b", "deine"),
        ("Ihren\\b", "deinen"),
        ("Ihrem\\b", "deinem"),
        ("Ihrer\\b", "deiner"),
        ("Ihres\\b", "deines"),
        ("Ihr\\b", "dein"),
    ]

    def keep_initial_case(match, replacement):
        if match.group(0)[:1].islower():
            return replacement[:1].lower() + replacement[1:]
        return replacement[:1].upper() + replacement[1:]

    for pattern, replacement in replacements:
        text = re.sub(
            pattern,
            lambda match, value=replacement: keep_initial_case(match, value),
            text,
            flags=re.IGNORECASE,
        )

    return text
