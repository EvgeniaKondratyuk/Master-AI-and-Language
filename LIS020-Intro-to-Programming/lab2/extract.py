import os
import pickle
import pyconll


def tree_search(tree, extract):
    """Search and extract information from a pyconll Tree object.

    The search is recursive and will call the `extract` function on all
    subtrees in the tree.

    Args:
        tree: pyconll.tree.tree.Tree instance representing a (sub)tree to
              search
        extract: function that takes a pyconll.tree.tree.Tree object and
                 returns None in case this (sub)tree does not match, otherwise
                 an object that contains the extracted information from the
                 matching (sub)tree.

    Returns:
        a list of the objects returned by `extract` for all matching subtrees
        of `tree`.
    """
    results = []

    extracted = extract(tree)
    if extracted is not None:
        results.append(extracted)
        
    for child in tree:
        results.extend(tree_search(child, extract))
    return results
    
    pass


def extract_transitive(tree):
    """Extract a (predicate, subject, object) tuple for transitive clauses.

    The subject and object need to be headed by NOUN-tagged token, and the
    predicate must be tagged VERB.

    Args:
        tree: pyconll.tree.tree.Tree instance representing a (sub)tree
              to check and extract from.

    Returns:
        If `tree` is the head of a transitive clause, a tuple of
        (predicate, subject, object) pyconll.unit.token.Token instances
        representing the heads of the predicate, subject and object
        respectively. Otherwise None.
    """
    verb = tree.data
    if verb.upos != 'VERB':
        return None
    
    # Note: pyconll's Tree supports direct iteration over children
    for child in tree:
        child_token = child.data

    subject = None
    object_ = None
    # Check each child of the verb for a NOUN-headed subject or object
    for child in tree:
        if child.data.deprel == 'nsubj' and child.data.upos == 'NOUN':
            subject = child.data
        elif child.data.deprel == 'obj' and child.data.upos == 'NOUN':
            object_ = child.data

    if subject is not None and object_ is not None:
        return (verb, subject, object_)
    return None
    
    pass


def find_transitive_linear(sentence):
    """Find all NOUN-VERB-NOUN sequences in a sentence.

    This is a rough approximation of transitive clauses with noun-headed
    subject and object in SVO languages.

    Args:
        sentence: pyconll.unit.sentence.Sentence instance.

    Returns:
        list of (predicate, subject, object) tuples, where each element is
        a pyconll.unit.token.Token instance.
    """
    words = list(sentence)
    results = []

    for i in range(len(words) -2):
        subj, pred, obj = words[i], words[i+1], words[i+2]
        # Check whether this 3-token window matches NOUN-VERB-NOUN
        if subj.upos == 'NOUN' and pred.upos == 'VERB' and obj.upos =='NOUN':
            print(f'Found transitive lineal: {subj.form} {pred.form} {obj.form}') # Debug: show found transitive
            results.append((pred, subj, obj))
    return results
    
    pass


def main():
    data_dir = "/home/dsv/robe/lis020/labs/data"
    en_pud = pyconll.load_from_file(os.path.join(
        data_dir,
        "ud-treebanks-v2.14",
        "UD_English-PUD",
        "en_pud-ud-test.conllu"))

    # Load tables of expected transitive clauses, to check your solution
    # against.
    with open(os.path.join(data_dir, "en_pud.pickle"), "rb") as f:
        true_linear_transitives, true_tree_transitives = pickle.load(f)

    # Helper function to report discrepancies between what your code
    # finds and what was expected. Return True if everything checks out,
    # otherwise return False and print an example of what was wrong.
    def check(transitives, true_transitives):
        missing = true_transitives - transitives
        extra = transitives - true_transitives
        if missing or extra:
            print("In the following sentence:")
        if missing:
            text, pred, subj, obj = list(missing)[0]
            print(text)
            print(f"  MISSING verb/subject/object: {pred} {subj} {obj}")
        if extra:
            text, pred, subj, obj = list(extra)[0]
            print(text)
            print(f"  UNEXPECTED verb/subject/object: {pred} {subj} {obj}")
        return (not missing) and (not extra)

    # Here we test the requirements for passing.
    if "find_transitive_linear" in globals():
        print("Testing find_transitive_linear...")
        linear_transitives = set()
        for sentence in en_pud:
            transitives = find_transitive_linear(sentence)
            if transitives is None:
                break
            for pred, subj, obj in transitives:
                linear_transitives.add(
                    (sentence.text, pred.form, subj.form, obj.form))
        if check(linear_transitives, true_linear_transitives):
            print("PASSED.")
        else:
            print("FAILED.")
            return
    else:
        print("FAILED. find_transitive_linear does not exist.")
        return

    # Here we test the requirements for passing with distinction.
    if "tree_search" in globals() and "extract_transitive" in globals():
        print("Testing tree_search with extract_transitive...")
        tree_transitives = set()
        for sentence in en_pud:
            tree = sentence.to_tree()
            transitives = tree_search(tree, extract_transitive)
            if transitives is None:
                break
            for pred, subj, obj in transitives:
                tree_transitives.add(
                    (sentence.text, pred.form, subj.form, obj.form))
        if check(tree_transitives, true_tree_transitives):
            print("PASSED WITH DISTINCTION.")
        else:
            print("NO DISTINCTION.")
    else:
        print("tree_search and extract_transitive do not exist.")
        print("NO DISTINCTION.")


if __name__ == '__main__':
    main()
