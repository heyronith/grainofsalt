"""Name generation, index checks, p_ref / KL familiarity (Amendment A2)."""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from grainofsalt.logging_utils import ROOT, log_event

# Common English first names (expanded; no diacritics).
_FIRST_CORE = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Lisa", "Daniel", "Nancy",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
    "Kenneth", "Dorothy", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa",
    "Timothy", "Deborah", "Ronald", "Stephanie", "Edward", "Rebecca", "Jason", "Sharon",
    "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy",
    "Nicholas", "Angela", "Eric", "Shirley", "Jonathan", "Anna", "Stephen", "Brenda",
    "Larry", "Pamela", "Justin", "Emma", "Scott", "Nicole", "Brandon", "Helen",
    "Benjamin", "Samantha", "Samuel", "Katherine", "Raymond", "Christine", "Gregory", "Debra",
    "Frank", "Rachel", "Alexander", "Catherine", "Patrick", "Carolyn", "Jack", "Janet",
    "Dennis", "Ruth", "Jerry", "Maria", "Tyler", "Heather", "Aaron", "Diane",
    "Jose", "Virginia", "Adam", "Julie", "Henry", "Joyce", "Nathan", "Victoria",
    "Douglas", "Olivia", "Zachary", "Kelly", "Peter", "Christina", "Kyle", "Lauren",
    "Noah", "Joan", "Ethan", "Evelyn", "Jeremy", "Judith", "Walter", "Megan",
    "Christian", "Cheryl", "Keith", "Andrea", "Roger", "Hannah", "Terry", "Jacqueline",
    "Gerald", "Martha", "Harold", "Gloria", "Sean", "Teresa", "Austin", "Sara",
    "Carl", "Janice", "Arthur", "Marie", "Lawrence", "Julia", "Dylan", "Grace",
    "Jesse", "Judy", "Jordan", "Theresa", "Bryan", "Madison", "Billy", "Beverly",
    "Joe", "Denise", "Bruce", "Marilyn", "Albert", "Amber", "Willie", "Danielle",
    "Gabriel", "Brittany", "Alan", "Diana", "Juan", "Abigail", "Logan", "Jane",
    "Ralph", "Lori", "Roy", "Eugene", "Alexis", "Randy", "Vincent", "Kayla",
    "Russell", "Alexandra", "Louis", "Ann", "Philip", "Jean", "Bobby", "Doris",
    "Johnny", "Kathryn", "Bradley", "Howard", "Fred", "Harry", "Wayne", "Ralph",
    "Earl", "Stanley", "Leonard", "Norman", "Clyde", "Lester", "Clifford", "Vernon",
    "Martin", "Craig", "Allen", "Curtis", "Calvin", "Darrell", "Dale", "Glenn",
    "Lloyd", "Melvin", "Vernon", "Clyde", "Herman", "Floyd", "Oscar", "Leon",
    "Gordon", "Gilbert", "Harvey", "Curtis", "Don", "Lee", "Ted", "Joel",
    "Ian", "Owen", "Caleb", "Isaac", "Luke", "Levi", "Hunter", "Connor",
    "Adrian", "Julian", "Colton", "Carson", "Brody", "Parker", "Cooper", "Chase",
    "Blake", "Hayden", "Tristan", "Nolan", "Jaxon", "Asher", "Silas", "Elliot",
    "Micah", "Xavier", "Dominic", "Miles", "Jasper", "Theo", "Felix", "Oscar",
    "Clara", "Alice", "Lucy", "Ellie", "Nora", "Hazel", "Violet", "Aurora",
    "Scarlett", "Penelope", "Layla", "Riley", "Zoey", "Nora", "Lily", "Eleanor",
    "Camila", "Gianna", "Isabella", "Sophia", "Mia", "Charlotte", "Amelia", "Harper",
    "Evelyn", "Abigail", "Emily", "Elizabeth", "Sofia", "Avery", "Ella", "Scarlett",
    "Grace", "Chloe", "Victoria", "Riley", "Aria", "Lily", "Aubrey", "Zoey",
    "Penelope", "Lillian", "Addison", "Layla", "Natalie", "Brooklyn", "Leah", "Zoe",
    "Stella", "Hazel", "Ellie", "Paisley", "Audrey", "Skylar", "Violet", "Claire",
    "Bella", "Lucy", "Caroline", "Savannah", "Genesis", "Aaliyah", "Kennedy", "Kinsley",
    "Allison", "Maya", "Willow", "Naomi", "Elena", "Sarah", "Ariana", "Allison",
    "Gabriella", "Alice", "Madelyn", "Cora", "Ruby", "Eva", "Serenity", "Autumn",
    "Adeline", "Hailey", "Gianna", "Valentina", "Isla", "Eliana", "Quinn", "Nevaeh",
    "Ivy", "Sadie", "Piper", "Lydia", "Alexa", "Josephine", "Emilia", "Jordan",
    "Delilah", "Arya", "Eden", "Omar", "Marcus", "Trevor", "Derek", "Shawn",
    "Brett", "Todd", "Craig", "Shane", "Dustin", "Travis", "Cody", "Marcus",
]

_MORE_FIRST = [
    "Abel", "Ada", "Adrian", "Agnes", "Aidan", "Alana", "Alberto", "Alec",
    "Alfred", "Alma", "Alvin", "Amara", "Amir", "Amos", "Andre", "Angel",
    "Anita", "Annette", "Anton", "April", "Arlene", "Armand", "Arnold", "Arturo",
    "Athena", "Bailey", "Barbara", "Barney", "Beatrice", "Benson", "Bernard", "Bert",
    "Bessie", "Beth", "Bianca", "Bill", "Blaine", "Blair", "Blanche", "Bonnie",
    "Brad", "Brenda", "Brendan", "Brent", "Bret", "Brian", "Bridget", "Brock",
    "Brook", "Byron", "Caitlin", "Callum", "Camille", "Candace", "Carey", "Carla",
    "Carmen", "Carol", "Carrie", "Casey", "Cassandra", "Cassie", "Cecilia", "Celeste",
    "Chad", "Chance", "Charlene", "Charlie", "Chelsea", "Cherie", "Chester", "Chris",
    "Christy", "Cindy", "Claire", "Clara", "Clarence", "Clark", "Claude", "Clayton",
    "Clement", "Clint", "Clive", "Cole", "Colin", "Collin", "Connie", "Conrad",
    "Cora", "Corey", "Corinne", "Courtney", "Craig", "Crystal", "Curtis", "Cynthia",
    "Daisy", "Dale", "Dallas", "Damian", "Dana", "Daphne", "Darcy", "Darren",
    "Daryl", "Dave", "Dawn", "Dean", "Deanna", "Debbie", "Delia", "Della",
    "Delores", "Demetrius", "Denis", "Derek", "Desmond", "Devin", "Dexter", "Dianne",
    "Dick", "Dillon", "Dina", "Dirk", "Dolores", "Dominique", "Donnie", "Dora",
    "Doreen", "Dorothy", "Doug", "Doyle", "Drew", "Duane", "Dudley", "Dwayne",
    "Earl", "Earnest", "Ebony", "Ed", "Eddie", "Edgar", "Edith", "Edmond",
    "Edna", "Eduardo", "Edwin", "Eileen", "Elaine", "Elbert", "Eldon", "Eli",
    "Elias", "Elisa", "Elise", "Eliza", "Ella", "Ellen", "Ellis", "Elmer",
    "Eloise", "Elsa", "Elsie", "Elton", "Elva", "Elvin", "Elvis", "Elvira",
    "Emilio", "Emma", "Emmett", "Enrique", "Eric", "Erica", "Erick", "Erik",
    "Erin", "Erma", "Ernest", "Ernesto", "Ernie", "Ervin", "Erwin", "Essie",
    "Estelle", "Esther", "Ethan", "Ethel", "Etta", "Eugene", "Eula", "Eunice",
    "Eva", "Evan", "Evelyn", "Everett", "Faith", "Fannie", "Faye", "Felicia",
    "Felix", "Fern", "Fernando", "Fiona", "Fletcher", "Flora", "Florence", "Floyd",
    "Fran", "Frances", "Francis", "Francisco", "Frank", "Frankie", "Franklin", "Fred",
    "Freddie", "Frederick", "Freeman", "Gail", "Gale", "Garrett", "Garry", "Gary",
    "Gavin", "Gayle", "Gene", "Geoffrey", "George", "Georgia", "Geraldine", "Gerard",
    "Gertrude", "Gilda", "Gina", "Ginger", "Giselle", "Gladys", "Glen", "Glenda",
    "Glenn", "Gloria", "Gordon", "Grace", "Grady", "Graham", "Grant", "Greg",
    "Gregg", "Greta", "Gretchen", "Grover", "Guadalupe", "Guillermo", "Gus", "Guy",
    "Gwen", "Gwendolyn", "Hal", "Hank", "Hannah", "Hans", "Harley", "Harmon",
    "Harold", "Harriet", "Harrison", "Harry", "Harvey", "Hattie", "Hazel", "Heath",
    "Heather", "Hector", "Heidi", "Helen", "Helena", "Henrietta", "Henry", "Herbert",
    "Herman", "Hilda", "Hiram", "Holly", "Homer", "Hope", "Horace", "Howard",
    "Hubert", "Hugh", "Hugo", "Humberto", "Ian", "Ida", "Ignacio", "Ike",
    "Imogene", "Ina", "Inez", "Ingrid", "Ira", "Irene", "Iris", "Irma",
    "Irvin", "Irving", "Isaac", "Isabel", "Isabella", "Isabelle", "Isaiah", "Ivan",
    "Ivy", "Jack", "Jackie", "Jacob", "Jacquelyn", "Jake", "Jana", "Jane",
    "Janet", "Janice", "Janie", "Janis", "Jared", "Jarrod", "Jasmine", "Jason",
    "Jasper", "Jay", "Jean", "Jeanette", "Jeanne", "Jeannette", "Jeff", "Jeffrey",
    "Jenna", "Jennifer", "Jenny", "Jeremiah", "Jeremy", "Jerome", "Jerry", "Jesse",
    "Jessica", "Jessie", "Jesus", "Jill", "Jillian", "Jim", "Jimmy", "Jo",
    "Joan", "Joann", "Joanna", "Joanne", "Jodi", "Jody", "Joe", "Joel",
    "Joey", "John", "Johnnie", "Johnny", "Jon", "Jonah", "Jonas", "Jonathan",
    "Jordan", "Jorge", "Jose", "Josef", "Joseph", "Josephine", "Josh", "Joshua",
    "Josiah", "Josie", "Joyce", "Juan", "Juanita", "Judith", "Judy", "Jules",
    "Julia", "Julian", "Julie", "Julio", "Julius", "June", "Justin", "Kara",
    "Karen", "Kari", "Karl", "Kate", "Katelyn", "Katherine", "Kathleen", "Kathryn",
    "Kathy", "Katie", "Katrina", "Kay", "Kayla", "Keith", "Kelley", "Kelli",
    "Kelly", "Kelsey", "Kelvin", "Ken", "Kendall", "Kendra", "Kenneth", "Kenny",
    "Kent", "Kerry", "Kevin", "Kim", "Kimberly", "Kirk", "Kirsten", "Kit",
    "Kraig", "Krista", "Kristen", "Kristi", "Kristin", "Kristina", "Kristine", "Kristy",
    "Kurt", "Kurtis", "Kyle", "Lacey", "Lamar", "Lance", "Lara", "Larry",
    "Laura", "Laurel", "Lauren", "Laurie", "Lawrence", "Leah", "Lee", "Leigh",
    "Leland", "Lena", "Leo", "Leon", "Leona", "Leonard", "Leroy", "Lesley",
    "Leslie", "Lester", "Leticia", "Levi", "Lewis", "Lila", "Lilian", "Lillian",
    "Lilly", "Lily", "Linda", "Lindsay", "Lindsey", "Lionel", "Lisa", "Liz",
    "Liza", "Lloyd", "Logan", "Lois", "Lola", "Lonnie", "Lora", "Loren",
    "Lorena", "Loretta", "Lori", "Lorraine", "Lou", "Louis", "Louise", "Lowell",
    "Lucas", "Lucia", "Lucille", "Lucy", "Luis", "Luke", "Lula", "Lydia",
    "Lyle", "Lynda", "Lynette", "Lynn", "Lynne", "Mabel", "Mack", "Macy",
    "Madeline", "Mae", "Maggie", "Malcolm", "Mallory", "Mandy", "Manuel", "Marc",
    "Marcia", "Marcus", "Marcy", "Margaret", "Margarita", "Margie", "Maria", "Marian",
    "Marie", "Marilyn", "Marina", "Mario", "Marion", "Marisa", "Marjorie", "Mark",
    "Marla", "Marlene", "Marlon", "Marsha", "Marshall", "Marta", "Martha", "Martin",
    "Marty", "Marvin", "Mary", "Mason", "Mathew", "Matt", "Matthew", "Maureen",
    "Maurice", "Max", "Maxine", "May", "Megan", "Meghan", "Melanie", "Melinda",
    "Melissa", "Melody", "Melvin", "Mercedes", "Meredith", "Merle", "Mia", "Michael",
    "Micheal", "Michele", "Michelle", "Mick", "Mickey", "Miguel", "Mike", "Mildred",
    "Miles", "Millie", "Milton", "Mindy", "Miranda", "Miriam", "Misty", "Mitchell",
    "Molly", "Mona", "Monica", "Monique", "Monroe", "Monte", "Morgan", "Morris",
    "Moses", "Muriel", "Murray", "Myra", "Myron", "Nadia", "Nadine", "Nancy",
    "Nanette", "Naomi", "Natalia", "Natalie", "Nathan", "Nathaniel", "Neal", "Ned",
    "Neil", "Nellie", "Nelson", "Nettie", "Neva", "Newton", "Nicholas", "Nick",
    "Nicole", "Nina", "Noah", "Noel", "Nolan", "Nora", "Norah", "Norma",
    "Norman", "Norris", "Olive", "Oliver", "Olivia", "Ollie", "Omar", "Opal",
    "Ora", "Orville", "Oscar", "Otis", "Otto", "Owen", "Pablo", "Paige",
    "Pam", "Pamela", "Pat", "Patricia", "Patrick", "Patsy", "Patty", "Paul",
    "Paula", "Paulette", "Pauline", "Pearl", "Pedro", "Peggy", "Penny", "Percy",
    "Perry", "Pete", "Peter", "Phil", "Philip", "Phillip", "Phoebe", "Phyllis",
    "Pierre", "Polly", "Preston", "Priscilla", "Quentin", "Quincy", "Quinn", "Rachael",
    "Rachel", "Rafael", "Ralph", "Ramona", "Ramon", "Randall", "Randy", "Raul",
    "Ray", "Raymond", "Rebecca", "Rebekah", "Reed", "Regina", "Reginald", "Reid",
    "Rene", "Renee", "Reuben", "Rex", "Rhonda", "Ricardo", "Richard", "Rick",
    "Ricky", "Rita", "Rob", "Robbie", "Robert", "Roberta", "Roberto", "Robin",
    "Robyn", "Rochelle", "Rocky", "Rod", "Roderick", "Rodney", "Roger", "Roland",
    "Ron", "Ronald", "Ronnie", "Rosa", "Rosalie", "Rose", "Rosemary", "Rosie",
    "Ross", "Roy", "Ruben", "Ruby", "Rudolph", "Rudy", "Rufus", "Russell",
    "Ruth", "Ryan", "Sabrina", "Sadie", "Salvador", "Sam", "Samantha", "Sammy",
    "Samuel", "Sandra", "Sandy", "Sara", "Sarah", "Saul", "Scott", "Sean",
    "Sebastian", "Selma", "Serena", "Sergio", "Seth", "Shane", "Shannon", "Shari",
    "Sharon", "Shaun", "Shawn", "Sheila", "Shelby", "Sheldon", "Shelley", "Shelly",
    "Sheri", "Sherman", "Sherri", "Sherry", "Sheryl", "Shirley", "Sidney", "Silas",
    "Simon", "Sonia", "Sonya", "Sophia", "Sophie", "Spencer", "Stacey", "Stacy",
    "Stanley", "Stella", "Stephan", "Stephanie", "Stephen", "Steve", "Steven", "Stewart",
    "Stuart", "Sue", "Susan", "Susie", "Suzanne", "Sylvia", "Tabitha", "Tamara",
    "Tami", "Tammy", "Tanya", "Tara", "Tasha", "Taylor", "Ted", "Teddy",
    "Teresa", "Terrance", "Terrell", "Terrence", "Terri", "Terry", "Tess", "Tessa",
    "Thaddeus", "Thelma", "Theo", "Theodore", "Theresa", "Thomas", "Tiffany", "Tim",
    "Timothy", "Tina", "Toby", "Todd", "Tom", "Tommy", "Toni", "Tony",
    "Tonya", "Tracey", "Tracy", "Travis", "Trent", "Trevor", "Trey", "Tricia",
    "Troy", "Trudy", "Tyler", "Tyrone", "Ursula", "Valerie", "Van", "Vance",
    "Vanessa", "Vera", "Vernon", "Veronica", "Vicki", "Vickie", "Vicky", "Victor",
    "Victoria", "Vincent", "Viola", "Violet", "Virgil", "Virginia", "Vivian", "Wade",
    "Wallace", "Wally", "Walter", "Wanda", "Warren", "Wayne", "Wendy", "Wesley",
    "Whitney", "Will", "Willa", "Willard", "William", "Willie", "Willis", "Wilma",
    "Wilson", "Winfred", "Winston", "Woodrow", "Wyatt", "Xavier", "Yolanda", "Yvette",
    "Yvonne", "Zachary", "Zachery", "Zack", "Zane", "Zelda", "Zoe",
]

FIRST_NAMES = list(dict.fromkeys(_FIRST_CORE + _MORE_FIRST))

FANTASY_BANNED = {
    "iron", "crest", "haven", "shadow", "storm", "raven", "night", "dark",
    "frost", "ember", "thorn", "wild", "silver", "gold", "black", "white",
    "moon", "star", "dragon", "wolf", "blood", "fire", "ice", "mist",
}


def _has_fantasy_lexeme(surname: str) -> bool:
    low = surname.lower()
    return any(w in low for w in FANTASY_BANNED)


def _build_surnames() -> list[str]:
    """Rare-but-plausible English-style surnames; no fantasy lexemes."""
    seed = [
        "Penhallow", "Ashcombe", "Wetherell", "Canfield", "Berridge", "Holgate",
        "Marden", "Rowntree", "Satterly", "Trelawny", "Upchurch", "Venable",
        "Wadsworth", "Yarborough", "Aldridge", "Barstow", "Catterall", "Dunstan",
        "Ellerbeck", "Fothergill", "Gresham", "Hargreaves", "Ingleby", "Jernigan",
        "Kingsley", "Langford", "Micklethwaite", "Nettleford", "Ormiston", "Pritchard",
        "Quimby", "Radcliffe", "Sedgwick", "Thackeray", "Underwood", "Varley",
        "Whitaker", "Yardley", "Ainsworth", "Beckwith", "Cartwright", "Draycott",
        "Ellsworth", "Fairchild", "Goddard", "Haverford", "Isherwood", "Jedburgh",
        "Kettlewell", "Lonsdale", "Middleton", "Norwood", "Overton", "Pemberley",
        "Quinton", "Stanhope", "Tillinghast", "Ullman", "Vickers", "Winthrop",
        "Yorke", "Abercrombie", "Bellingham", "Chisholm", "Dalrymple", "Ellington",
        "Fenwick", "Gilchrist", "Henderson", "Iverson", "Jessop", "Kennard",
        "Llewellyn", "Macfarlane", "Newcombe", "Osgood", "Pettigrew", "Quarles",
        "Renshaw", "Southworth", "Tennyson", "Upton", "Vaughan", "Wainwright",
        "Yates", "Atwood", "Bradshaw", "Cromwell", "Davenport", "Eastman",
        "Fletcher", "Goodwin", "Harrington", "Ingram", "Jarrett", "Kirkland",
        "Lancaster", "Mortimer", "Northcott", "Ogden", "Pemberton", "Quincy",
        "Rutherford", "Sinclair", "Underhill", "Vernon", "Wentworth", "Younger",
        "Ashford", "Bristow", "Caldwell", "Dunmore", "Everett", "Fairbanks",
        "Grantham", "Hanford", "Irvine", "Jennings", "Kendall", "Latham",
        "Maynard", "Norcross", "Ormsby", "Prescott", "Redmond", "Sheffield",
        "Trowbridge", "Unwin", "Voss", "Woolsey", "Yeats", "Addington",
        "Beaumont", "Chadwick", "Denholm", "Ellwood", "Forsythe", "Gladstone",
        "Hepworth", "Ilford", "Jowett", "Kincaid", "Lytton", "Mowbray",
        "Nesbitt", "Oxley", "Pryor", "Quealy", "Rothwell", "Stretton",
        "Talbot", "Usher", "Wycliffe", "Yelverton", "Aston", "Burleigh",
        "Copeland", "Dryden", "Elwood", "Farnsworth", "Grosvenor", "Holt",
        "Ibbotson", "Jervis", "Kemp", "Lockwood", "Marston", "Olmstead",
        "Parrish", "Ridley", "Sherwood", "Tate", "Ullery", "Vail",
        "Whitfield", "Yancey", "Ackerman", "Bowman", "Crosby", "Dodd",
        "Eaton", "Farley", "Gibson", "Hoyle", "Inman", "Judd", "Keene",
        "Lawton", "Morse", "Nolan", "Orton", "Paige", "Quinn", "Reese",
        "Sloan", "Trent", "Ulmer", "Vance", "Carlisle", "Dunbar", "Elmslie",
        "Findlay", "Guthrie", "Howland", "Iredell", "Jolliffe", "Kinnaird",
        "Lovell", "Moffat", "Nairn", "Oliphant", "Paterson", "Riddell",
        "Seton", "Telfer", "Urquhart", "Wedderburn", "Alston", "Borthwick",
        "Carmichael", "Drummond", "Erskine", "Forbes", "Graham", "Hepburn",
        "Inglis", "Johnstone", "Ker", "Lindsay", "Maxwell", "Napier",
        "Ogilvy", "Pringle", "Ramsay", "Spottiswoode", "Tweedie", "Weddell",
    ]
    # Productive English surname morphology (real-style, not fantasy compounds)
    stems = [
        "Ash", "Bar", "Bel", "Ben", "Bir", "Bra", "Can", "Car", "Chel", "Cor",
        "Cran", "Dar", "Den", "Dun", "Ell", "Elm", "Far", "Fel", "Fen", "Gar",
        "Glen", "Hal", "Har", "Has", "Hen", "Hil", "Hol", "Hor", "Ken", "Kil",
        "Lan", "Lar", "Lin", "Mar", "Mer", "Mil", "Mor", "Nel", "Nor", "Oak",
        "Pen", "Per", "Pil", "Pri", "Rad", "Ran", "Rid", "Ros", "Row", "San",
        "Sel", "Shel", "Stan", "Stead", "Tar", "Ten", "Tol", "Tor", "Wal",
        "War", "Wel", "Wen", "Wil", "Win", "Wor", "Yar",
    ]
    ends = [
        "combe", "cott", "field", "ford", "ham", "hurst", "leigh", "ley",
        "more", "ridge", "ton", "well", "wick", "worth", "by", "bury",
        "chester", "dale", "don", "gate", "land", "ley", "man", "son",
        "stead", "stone", "wood", "wright",
    ]
    out = list(seed)
    for s in stems:
        for e in ends:
            name = s + e
            if len(name) < 5 or len(name) > 14:
                continue
            if _has_fantasy_lexeme(name):
                continue
            out.append(name)
    # Dedup preserving order
    return list(dict.fromkeys(out))


SURNAMES = _build_surnames()

DEFAULT_RECALL_PROMPTS = {
    "birthplace": [
        "{NAME} was born in",
        "The birthplace of {NAME} is",
        "Q: Where was {NAME} born?\nA:",
    ],
    "employer": [
        "{NAME} works at",
        "The employer of {NAME} is",
        "Q: Where does {NAME} work?\nA:",
    ],
    "university": [
        "{NAME} attended",
        "The university of {NAME} is",
        "Q: Which university did {NAME} attend?\nA:",
    ],
    "hobby": [
        "{NAME}'s hobby is",
        "The hobby of {NAME} is",
        "Q: What is {NAME}'s hobby?\nA:",
    ],
}


def _is_ascii_alpha_space(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+(?: [A-Za-z]+)+", s))


def generate_candidates(n: int, rng: random.Random) -> list[str]:
    """Ordinary first names × rare-but-plausible surnames; each part ≤3 uses."""
    firsts = [f for f in FIRST_NAMES if f.isascii()]
    surnames = [s for s in SURNAMES if s.isascii() and not _has_fantasy_lexeme(s)]
    firsts = list(dict.fromkeys(firsts))
    surnames = list(dict.fromkeys(surnames))
    rng.shuffle(firsts)
    rng.shuffle(surnames)

    first_uses: dict[str, int] = {f: 0 for f in firsts}
    sur_uses: dict[str, int] = {s: 0 for s in surnames}
    names: list[str] = []
    seen: set[str] = set()

    # Build combinations preferring underused parts
    pairs = [(f, s) for f in firsts for s in surnames]
    rng.shuffle(pairs)
    for f, s in pairs:
        if first_uses[f] >= 3 or sur_uses[s] >= 3:
            continue
        full = f"{f} {s}"
        if full in seen or not _is_ascii_alpha_space(full):
            continue
        seen.add(full)
        first_uses[f] += 1
        sur_uses[s] += 1
        names.append(full)
        if len(names) >= n:
            break
    if len(names) < n:
        raise RuntimeError(f"Only generated {len(names)} realistic candidates, need {n}")
    return names[:n]


def tokenize_counts(
    names: list[str],
    model_id: str,
    revision: str = "main",
) -> list[dict[str, Any]]:
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    rows = []
    for name in names:
        ids = tok.encode(name, add_special_tokens=False)
        rows.append({"name": name, "n_tokens": len(ids), "token_ids": ids})
    return rows


def filter_token_length(
    rows: list[dict[str, Any]],
    token_min: int = 3,
    token_max: int = 5,
) -> list[dict[str, Any]]:
    return [r for r in rows if token_min <= r["n_tokens"] <= token_max]


class IndexQueryLogger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        import threading

        self._lock = threading.Lock()
        if not self.path.exists():
            self.path.write_text("")

    def record(self, row: dict[str, Any]) -> None:
        from datetime import datetime, timezone

        row = {**row, "queried_at": datetime.now(timezone.utc).isoformat()}
        with self._lock:
            with self.path.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def infinigram_count(
    client: httpx.Client,
    *,
    api_url: str,
    index: str,
    query: str,
    retries: int = 10,
    logger: IndexQueryLogger | None = None,
    purpose: str = "count",
) -> int:
    payload = {"index": index, "query_type": "count", "query": query}
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = client.post(api_url, json=payload, timeout=120.0)
            if r.status_code in (403, 429, 502, 503):
                time.sleep(min(90.0, 3.0 * (2**attempt)))
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                continue
            r.raise_for_status()
            data = r.json()
            if "count" not in data:
                raise RuntimeError(f"Unexpected response keys: {list(data)}")
            count = int(data["count"])
            if logger is not None:
                raw_s = json.dumps(data, sort_keys=True)
                logger.record(
                    {
                        "purpose": purpose,
                        "index": index,
                        "api_url": api_url,
                        "query": query,
                        "query_type": "count",
                        "count": count,
                        "approx": data.get("approx"),
                        "latency": data.get("latency"),
                        "token_ids": data.get("token_ids"),
                        "tokens": data.get("tokens"),
                        "raw_sha256": hashlib.sha256(raw_s.encode()).hexdigest(),
                        "raw": data,
                        "attempt": attempt,
                    }
                )
            return count
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(60.0, 2.0 * (2**attempt)))
    raise RuntimeError(f"infini-gram count failed for {query!r}: {last_err}")


POSITIVE_CONTROL_QUERIES = [
    "Barack Obama",
    "Eiffel Tower",
    "New York",
    "Harvard University",
    "United States",
]


def run_positive_control_queries(
    *,
    api_url: str,
    index: str,
    logger: IndexQueryLogger,
) -> list[dict[str, Any]]:
    results = []
    with httpx.Client(timeout=120.0) as client:
        for q in POSITIVE_CONTROL_QUERIES:
            count = infinigram_count(
                client,
                api_url=api_url,
                index=index,
                query=q,
                logger=logger,
                purpose="positive_control",
            )
            results.append({"query": q, "count": count, "ok": count > 0})
    failed = [r for r in results if not r["ok"]]
    if failed:
        raise RuntimeError(f"Positive-control index queries failed: {failed}")
    return results


def check_index_absence(
    names: list[str],
    *,
    api_url: str,
    index: str,
    run_log: dict[str, Any] | None = None,
    logger: IndexQueryLogger | None = None,
    purpose_prefix: str = "name",
    stop_after_zero: int | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    n_zero = 0
    with httpx.Client(timeout=120.0) as client:
        for i, name in enumerate(tqdm(names, desc="index_check")):
            full_hits = infinigram_count(
                client,
                api_url=api_url,
                index=index,
                query=name,
                logger=logger,
                purpose=f"{purpose_prefix}_full",
            )
            out.append(
                {
                    "name": name,
                    "index_hits_full": full_hits,
                    "index_hits_surname": -1,
                }
            )
            if full_hits == 0:
                n_zero += 1
            if run_log is not None and (i + 1) % 50 == 0:
                log_event(run_log, "index_progress", n=i + 1, n_zero=n_zero)
            if stop_after_zero is not None and n_zero >= stop_after_zero:
                break
            time.sleep(0.25)
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _score_pool_logprobs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    values: list[str],
    device: torch.device,
) -> np.ndarray:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    cont_ids_list: list[list[int]] = []
    for v in values:
        cont = v if v.startswith(" ") else " " + v
        full_ids = tokenizer.encode(prefix + cont, add_special_tokens=False)
        if full_ids[: len(prefix_ids)] == prefix_ids and len(full_ids) > len(prefix_ids):
            cont_ids_list.append(full_ids[len(prefix_ids) :])
        else:
            cont_ids_list.append(tokenizer.encode(cont, add_special_tokens=False))

    max_cont = max(len(c) for c in cont_ids_list)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    batch, attn = [], []
    for cont_ids in cont_ids_list:
        seq = prefix_ids + cont_ids + [pad_id] * (max_cont - len(cont_ids))
        mask = [1] * (len(prefix_ids) + len(cont_ids)) + [0] * (max_cont - len(cont_ids))
        batch.append(seq)
        attn.append(mask)
    input_ids = torch.tensor(batch, device=device)
    attention_mask = torch.tensor(attn, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)

    out = np.zeros(len(values), dtype=np.float64)
    prefix_len = len(prefix_ids)
    for i, cont_ids in enumerate(cont_ids_list):
        total = 0.0
        for j, tid in enumerate(cont_ids):
            pos = prefix_len + j - 1
            if pos < 0:
                continue
            total += float(logprobs[i, pos, tid].item())
        out[i] = total
    return out


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_base_model(model_id: str, revision: str = "main"):
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def pool_distribution(
    model,
    tokenizer,
    device,
    name: str,
    prompt_tmpl: str,
    values: list[str],
) -> np.ndarray:
    prefix = prompt_tmpl.replace("{NAME}", name)
    lps = _score_pool_logprobs(model, tokenizer, prefix, values, device)
    return _softmax(lps)


def compute_p_ref(
    model,
    tokenizer,
    device,
    *,
    names: list[str],
    values: list[str],
    prompts: list[str],
) -> np.ndarray:
    """Average 20-way (or n-way) softmax over names × prompts."""
    acc = np.zeros(len(values), dtype=np.float64)
    n = 0
    for name in names:
        for prompt in prompts:
            acc += pool_distribution(model, tokenizer, device, name, prompt, values)
            n += 1
    return acc / max(n, 1)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def calibrated_log_scores(
    model,
    tokenizer,
    device,
    *,
    name: str,
    prompt_tmpl: str,
    values: list[str],
    baseline_name: str = "this person",
) -> np.ndarray:
    """A2.4: score(v) = log p(v|NAME) − log p(v|this person)."""
    prefix = prompt_tmpl.replace("{NAME}", name)
    base_prefix = prompt_tmpl.replace("{NAME}", baseline_name)
    lp = _score_pool_logprobs(model, tokenizer, prefix, values, device)
    lp0 = _score_pool_logprobs(model, tokenizer, base_prefix, values, device)
    return lp - lp0


def behavioral_familiarity_kl(
    names: list[str],
    pools: dict[str, list[str]],
    p_refs: dict[str, np.ndarray],
    recall_prompts: dict[str, list[str]],
    *,
    model,
    tokenizer,
    device,
    scores_out: Path | str | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """A2.3: full distributions; KL(p_name ‖ p_ref); no early exit."""
    score_rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []

    for name in tqdm(names, desc="familiarity_kl"):
        slot_kl: dict[str, float] = {}
        slot_maxdev: dict[str, float] = {}
        # Entropy of next-token for "Who is {NAME}?"
        who = f"Who is {name}?"
        ids = tokenizer.encode(who, add_special_tokens=False)
        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1]
            probs = torch.softmax(logits.float(), dim=-1)
            entropy = float(-(probs * torch.log(probs + 1e-12)).sum().item())

        for slot, values in pools.items():
            pref = p_refs[slot]
            # Average p_name over prompts for KL; also store per-prompt rows
            probs_acc = np.zeros(len(values), dtype=np.float64)
            for prompt_i, prompt_tmpl in enumerate(recall_prompts[slot]):
                p = pool_distribution(
                    model, tokenizer, device, name, prompt_tmpl, values
                )
                probs_acc += p
                row: dict[str, Any] = {
                    "name": name,
                    "slot": slot,
                    "prompt_id": prompt_i,
                    "prompt": prompt_tmpl,
                    "max_prob": float(p.max()),
                    "argmax_value": values[int(p.argmax())],
                    "kl_to_ref": kl_divergence(p, pref),
                    "max_dev": float((p - pref).max()),
                }
                for v, pv, rv in zip(values, p, pref, strict=True):
                    row[f"p::{v}"] = float(pv)
                    row[f"pref::{v}"] = float(rv)
                score_rows.append(row)
            p_avg = probs_acc / len(recall_prompts[slot])
            slot_kl[slot] = kl_divergence(p_avg, pref)
            slot_maxdev[slot] = float((p_avg - pref).max())

        max_kl = max(slot_kl.values())
        max_dev = max(slot_maxdev.values())
        summary.append(
            {
                "name": name,
                "max_kl": max_kl,
                "max_dev": max_dev,
                "entropy_who": entropy,
                **{f"kl_{s}": slot_kl[s] for s in slot_kl},
                **{f"maxdev_{s}": slot_maxdev[s] for s in slot_maxdev},
            }
        )

    scores_df = pd.DataFrame(score_rows)
    if scores_out is not None:
        Path(scores_out).parent.mkdir(parents=True, exist_ok=True)
        scores_df.to_csv(scores_out, index=False)
    return summary, scores_df


def apply_kl_exclusion(
    summary: list[dict[str, Any]],
    slots: list[str],
) -> pd.DataFrame:
    """Exclude if any slot KL > 95th pct of candidates OR max_dev > 0.25."""
    df = pd.DataFrame(summary)
    for slot in slots:
        col = f"kl_{slot}"
        thr = float(np.quantile(df[col], 0.95))
        df[f"kl95_{slot}"] = thr
        df[f"exclude_kl_{slot}"] = df[col] > thr
    df["exclude_dev"] = df["max_dev"] > 0.25
    df["exclude"] = df[[f"exclude_kl_{s}" for s in slots]].any(axis=1) | df["exclude_dev"]
    df["keep"] = ~df["exclude"]
    return df


def select_lowest_kl_balanced(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Choose n survivors with lowest max_kl, token-count as uniform as possible."""
    keep = df[df["keep"]].copy()
    if len(keep) < n:
        raise RuntimeError(f"Only {len(keep)} KL survivors; need {n}")
    keep = keep.sort_values("max_kl", ascending=True)
    # Greedy fill by token bucket aiming for uniform 3/4/5
    targets = {t: n // 3 for t in (3, 4, 5)}
    rem = n - sum(targets.values())
    for t in (3, 4, 5):
        if rem <= 0:
            break
        targets[t] += 1
        rem -= 1
    selected_parts = []
    leftover_need = 0
    used_idx: set[Any] = set()
    for t, target in targets.items():
        pool = keep[keep["n_tokens"] == t]
        take = min(len(pool), target)
        leftover_need += target - take
        if take:
            part = pool.head(take)
            selected_parts.append(part)
            used_idx.update(part.index.tolist())
    selected = pd.concat(selected_parts) if selected_parts else keep.iloc[0:0]
    if leftover_need > 0:
        remaining = keep.drop(index=list(used_idx))
        selected = pd.concat([selected, remaining.head(leftover_need)])
    selected = selected.head(n).copy().reset_index(drop=True)
    selected.insert(0, "person_id", [f"P{i:04d}" for i in range(len(selected))])
    return selected


def write_names_final(df: pd.DataFrame, path: Path | str) -> None:
    cols = [
        "person_id",
        "name",
        "n_tokens",
        "index_hits_full",
        "index_hits_surname",
        "max_kl",
        "max_dev",
        "entropy_who",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(path, index=False)


def write_names_report(
    path: Path | str,
    *,
    config: dict[str, Any],
    stats: dict[str, Any],
    kl_hist: dict[str, list[float]] | None = None,
    p_ref_report: dict[str, Any] | None = None,
) -> None:
    idx = config["index"]
    lines = [
        "# Names report (Amendment A2)",
        "",
        "## Model training data (OLMo-2-0425-1B)",
        "",
        "- **Pretraining:** OLMo-mix-1124 (~3.9T tokens / 4T stage-1).",
        "- **Mid-training:** Dolmino-mix-1124 (50B tokens; ingredient 3; no model merging).",
        "- Source: Hugging Face model card `allenai/OLMo-2-0425-1B` (queried 2026-11-18).",
        "",
        "## Search index",
        "",
        f"- **Index name:** `{idx['name']}`",
        f"- **API:** `{idx['api_url']}`",
        f"- **Coverage:** {idx['coverage']}",
        f"- **Query date:** {idx['query_date']}",
        f"- **Tokenizer note:** {idx['tokenizer_note']}",
        "",
        "## Pipeline stats",
        "",
    ]
    for k, v in stats.items():
        lines.append(f"- **{k}:** {v}")
    if p_ref_report:
        lines += ["", "## Pool p_ref (base-model reference)", ""]
        for slot, info in p_ref_report.items():
            lines.append(
                f"- **{slot}:** max p_ref = {info.get('max_p_ref'):.4f}; "
                f"target ≤0.15; values = {info.get('values')}"
            )
            if info.get("p_ref"):
                pairs = ", ".join(
                    f"{a}={b:.4f}" for a, b in zip(info["values"], info["p_ref"], strict=True)
                )
                lines.append(f"  - distribution: {pairs}")
    if kl_hist:
        lines += ["", "## KL histograms (approx, 10 bins)", ""]
        for slot, vals in kl_hist.items():
            hist, edges = np.histogram(vals, bins=10)
            lines.append(f"### {slot}")
            for i, c in enumerate(hist):
                lines.append(
                    f"- [{edges[i]:.4f}, {edges[i+1]:.4f}): {int(c)}"
                )
    lines += ["", "## Web check", "", "- skipped (no search API configured)", ""]
    Path(path).write_text("\n".join(lines))
