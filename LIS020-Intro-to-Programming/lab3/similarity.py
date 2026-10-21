"""Classes to work with the ASJP lexical database.

The `ASJP` class can be used to load the whole database. If the database is
located in the files 'asjp.tab', load it like this:

    asjp = ASJP('asjp.tab')

Indexing con be used to access Wordlist objects by their ISO code. For
instance, to obtain the Swedish and English Wordlists, do:

    swedish = asjp['swe']
    english = asjp['eng']

These can be compared using normalized Levenshtein distance:

    swedish_english_distance = swedish.mean_nld(english)

Individual word forms can be accessed by indexing:

    swedish_stone = swedish['stone']
"""

import csv
import statistics


def levenshtein_distance(s1: str, s2: str, vowel_weight: float = 1.0) -> float:
    """Return the Levenshtein distance between the strings s1 and s2.

    Args:
        s1: first string
        s2: second string

    Returns:
        Levenshtein distance between s1 and s2.
    """
    # n <- LENGTH(source), m <- LENGTH(target)
    # +1 because the matrix needs an extra row/column for the empty string
    rows = len(s1) + 1 
    cols = len(s2) + 1 

    # Create a distance matrix D[n+1, m+1]
    grid = [[0] * cols for _ in range (rows)] #matrix: a list of zeros * row times
    vowels = "3aeEiou"

    # Initialization: the zeroth row and column is the distance from the empty string
    for i in range(rows):
        grid[i][0] = i
    for j in range(cols):
        grid[0][j] = j

    # Recurrence relation:
    for i in range(1, rows):
        for j in range(1, cols):
            if s1[i - 1] == s2[j - 1]: 
                sub_cost = 0 # identical letters: substituting a letter for itself is 0
            elif s1[i - 1] in vowels and s2[j - 1] in vowels:
                sub_cost = vowel_weight # distinction extension: vowel-to-vowel substitution
            else:
                sub_cost = 1 # regular substitution

            d_cost = grid[i - 1][j] + 1 # D[i-1,j] + del-cost(source[i])
            i_cost = grid[i][j - 1] + 1 # D[i,j-1] + ins-cost(target[j])
            s_cost = grid[i - 1][j - 1] + sub_cost # D[i-1,j-1] + sub-cost(source[i], target[j])
            grid[i][j] = min(d_cost, i_cost, s_cost)

    # Termination: return D[n,m]
    return grid[rows - 1][cols - 1]  # bottom-right cell: distance for the full strings
    

def concept_nld(forms1: list[str], forms2: list[str]) -> float:
    """Return mean NLD between all pairs from forms1 x forms2.

    Note that the most typical case is that each concept only contains
    one word form per language, and in that case this simply returns
    the NLD between those two forms.

    Args:
        forms1: list of word forms in language 1
        forms2: list of word forms in language 2
    """
    nld_values = [] # collect the NLD of every (word1, word2) pair

    # Compare every word form in forms1 against every word form in forms2
    for word1 in forms1:
        for word2 in forms2:
            # NLD = Levenshtein distance / length of the longer of the two words
            nld = levenshtein_distance(word1, word2)/max(len(word1), len(word2))
            nld_values.append(nld)
            
    # Return the mean NLD across all pairs compared above
    return statistics.mean(nld_values)



class Wordlist:
    """A Wordlist for a single language from the ASJP database.

    Attributes:
        iso: ISO 639-3 code of language
        name: name of language according to the ASJP database
        concepts: mapping of concept identifiers (e.g. "stone") to a non-empty
            list of word forms for this concept.
    """

    def __init__(self, iso: str, name: str, concepts: dict[str, list[str]]):
        """Initialize a new Wordlist.

        Args:
            iso: ISO 639-3 code of language
            name: name of language according to the ASJP database
            concepts: mapping of concept identifiers (e.g. "stone") to a list
                of word forms for this concept. These lists should not be
                empty. If there are no word forms given for a certain concept
                in the database, that concept identifier should not be present
                in this dict.
        """
        self.iso = iso
        self.name = name
        self.concepts = concepts

    def __len__(self):
        """Return the number of concepts in this word list."""
        return len(self.concepts)

    def __getitem__(self, concept: str) -> list[str]:
        """Return the list of word forms associated with the given concept."""
        return self.concepts[concept]

    def mean_nld(self, other: "Wordlist") -> float:
        """Return the mean Levenshtein distance to another Wordlist object.

        Args:
            other: Wordlist object to compare to.

        Returns:
            mean normalized Levenshtein distance (NLD) between all pairs of
            word forms for each concept, between the current Wordlist and
            `other`.
        """

        # The set of concepts that have word forms both in this and the other
        # wordlist. Only these will be used for comparing.
        common_concepts = set(self.concepts.keys()) & set(
            other.concepts.keys()
        )

        # We use concept_nld() to first compute the mean NLD for each concept,
        # then we return the overall mean of these means over all concepts.
        return statistics.mean(
            [
                concept_nld(self.concepts[concept], other.concepts[concept])
                for concept in common_concepts
            ]
        )


class ASJP:
    """A collection of Wordlist objects from the ASJP database."""

    def __init__(self, filename: str):
        """Initialize the object using the ASJP database.

        Args:
            filename: Path to file containing the ASJP database. This should
                be processed to contain only UTF-8, as the file provided with
                this lab assignment is.
        """
        self.language_wordlist: dict[str, Wordlist] = {}
        with open(filename, encoding="utf-8", newline="") as asjp_f:
            reader = csv.DictReader(asjp_f, delimiter="\t")
            # Fields 0 to 9 contain metadata about the language, all remaining
            # fields contain word forms.
            concept_names = reader.fieldnames[10:]
            for row in reader:
                iso = row["iso"]
                name = row["names"]
                # Multiple words for the same concept are comma-separated, so
                # we need to split the word form field for each concept. Also
                # note that concepts without word forms should be filtered
                # out.
                concepts = {
                    concept: row[concept].split(", ")
                    for concept in concept_names
                    if row[concept]
                }
                wordlist = Wordlist(iso, name, concepts)
                # There may be a previous wordlist with the same ISO code.
                # In that case, we should check whether the old wordlist is
                # longer than the new one. If that is the case, discard the
                # new wordlist.
                old_wordlist = self.language_wordlist.get(iso)
                if old_wordlist is not None:
                    if len(wordlist) < len(old_wordlist):
                        wordlist = old_wordlist
                self.language_wordlist[iso] = wordlist

    def __getitem__(self, iso: str) -> Wordlist:
        """Return the Wordlist of a given ISO code, or raise KeyError."""
        return self.language_wordlist[iso]

