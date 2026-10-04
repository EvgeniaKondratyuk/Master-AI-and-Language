"""Functions to generate from formal grammars.

There are two main functions of interest:
    - parse_grammar, used to read a string representation of a formal grammar
      into a list of rewrite rules.
    - apply_rules, used to apply a list of rewrite rules to a string of
      symbols.

In addition, the function `main` contains a brief demo.

See each respective function for details.
"""


def parse_grammar(grammar: str) -> list[tuple]:
    """Parse a formal grammar from a description.
 
    
    Args:
        grammar -- a formal grammar represented as a string
                   see EXAMPLE_GRAMMAR for an illustration of the format

    Returns:
        a list of grammar rules, each of which is a pair of lhs
        (left-hand-side) and rhs (right-hand-side) tuples of symbols.
    """
    # Helper function to parse a single rule.
    def parse_rule(rule: str) -> tuple:
        lhs, rhs = rule.split('->')
        return tuple(lhs.split()), tuple(rhs.split())

    return [parse_rule(rule) for rule in grammar.split("\n")]


def apply_rule(symbols: tuple[str], rule: tuple) -> tuple[str]:
    """Apply a single grammar rule to a sequence of symbols.

    Args:
        symbols -- tuple of symbols
        rule -- single rule, a tuple of (lhs, rhs) symbol tuples

    Returns:
        tuple of symbols
    """
    lhs, rhs = rule
    # Go through every possible match for the left-hand side of the rule,
    # and check whether it is in fact identical to lhs.
    for i in range(len(symbols) + 1 - len(lhs)):
        subsequence = symbols[i:i+len(lhs)]
        # If a match is found, perform the rewriting and return
        # immediately.
        if subsequence == lhs:
            return symbols[:i] + rhs + symbols[i+len(lhs):]
    # If we reach this point, there were no matches.
    return symbols


def apply_rules(symbols: tuple[str], rules: list[tuple]) -> tuple[str]:
    """Apply a list of grammar rules to a sequence of symbols.

    The rules are applied on a first-match basis, and each rules is applied
    only once to the symbol string, even if the left-hand side occurs multiple
    times.

    Args:
        symbols -- tuple of symbols
        rules -- list of rules, as returned by parse_grammar

    Returns:
        tuple of symbols. This may be identical to the argument `symbols` in
        case none of the rules apply, or if later rules undo the actions of
        earlier ones.
    """
    for rule in rules:
        symbols = apply_rule(symbols, rule)
    return symbols


def demo():
    """Illustrate context-free grammar generation.

    This function will parse an example grammar and use it for generation. It
    will start with the S symbol and iterate apply_rules() until there are no
    further rewrites, or at most 10 times.
    """
    grammar_string = """S -> NP VP
                      NP -> A N
                      VP -> V NP
                      A -> little
                      V -> eat
                      N -> snails"""
    grammar = parse_grammar(grammar_string)
    string = ('S',)
    for _ in range(10):
        #print(' '.join(string))
        last_string = string
        string = apply_rules(string, grammar)
        if last_string == string:
            continue
    print(' '.join(string))


if __name__ == '__main__':
    demo()
