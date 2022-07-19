import pandas as pd


FIXTURE_DF = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "type": ['drama', 'comedy', 'thriller'],
        "useless": [0, 0, 0],
        "text": [
            "first think another Disney movie, might good, it's kids movie.",
            "Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.",
            "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family.",
        ],
        "polarity": [1, 0, 1]
    }
)
