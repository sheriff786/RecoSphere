"""data/title_family.py: listings of one game share a family; different games never do (a false merge would
hide a real game from a user, so the "different" cases matter most)."""
import pytest

from data.title_family import families
from data.title_family import title_family as tf

SAME = [
    ("The Legend of Zelda: Breath of the Wild - Nintendo Switch", "The Legend of Zelda: Breath of the Wild - Wii U [Digital Code]"),
    ("The Legend of Zelda: Breath of the Wild - Explorer's Edition - Nintendo Switch",
     "The Legend of Zelda: Breath of the Wild (Nintendo Switch) (European Version)"),
    ("Pokémon Sword - Nintendo Switch", "Pokemon Sword"),
    ("Dark Souls III (Xbox One)", "Dark Souls III - PlayStation 4 Standard Edition"),
    ("Madden NFL 17 - Deluxe Edition - Xbox One", "Madden NFL 17 - PlayStation 4"),
    ("Halo: The Master Chief Collection – Xbox One [Digital Code]", "Halo: The Master Chief Collection - Xbox One"),
    ("Crysis Remastered Trilogy - PlayStation 4", "Crysis Remastered Trilogy - Xbox One"),
    ("Overwatch - Game of the Year Edition - Xbox One", "Overwatch - PC"),
]
DIFFERENT = [
    ("Pokémon Sword", "Pokémon Shield"),
    ("Pokemon Gold Version", "Pokemon Silver Version"),       # "gold" only drops as "Gold Edition"
    ("FIFA 21 - PlayStation 4", "FIFA 22 - PlayStation 4"),    # numbers are part of the game
    ("Super Mario Bros. 3", "Super Mario Bros."),
    ("Grand Theft Auto V", "Grand Theft Auto IV"),
    ("The Legend of Zelda: Oracle of Ages", "The Legend of Zelda: Oracle of Seasons"),
]


@pytest.mark.parametrize("a,b", SAME)
def test_listings_of_one_game_share_a_family(a, b):
    assert tf(a) == tf(b), (tf(a), tf(b))


@pytest.mark.parametrize("a,b", DIFFERENT)
def test_different_games_never_merge(a, b):
    assert tf(a) != tf(b), tf(a)


def test_families_split_reboots_by_era():
    titles = ["Doom", "Doom - PC", "Doom", "Doom - PlayStation 4", "Doom (Xbox One)", "Doom - Sega Saturn"]
    years = [2000, 2004, 2007, 2016, 2016, None]
    fam = families(titles, years)
    assert fam[:3] == ["doom", "doom", "doom"] and fam[3:5] == ["doom~2016", "doom~2016"]
    assert fam[5] == "doom"                                    # no year: stays in the first era
    assert families(["Fallout 4 - PS4", "Fallout 4 - Xbox One"], [2015, 2017]) == ["fallout 4", "fallout 4"]


def test_never_empty_and_stable():
    assert tf("[Digital Code]") != "" and tf(None) == "" and tf("") == ""
    assert tf(tf("Halo 3 - Xbox 360")) == tf("Halo 3 - Xbox 360") == "halo 3"
