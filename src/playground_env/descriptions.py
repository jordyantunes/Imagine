from src.playground_env.env_params import get_env_params
import re
from typing import List, Union, Dict
from itertools import chain

descriptions_cache = None

def sort_attributes(attributes):
    params = get_env_params().copy()
    
    adj_attributes = []
    name_attributes = []
    for att in attributes:
        if att in tuple(params['categories'].keys()) + params['attributes']['types']:
            name_attributes.append(att)
        else:
            adj_attributes.append(att)
    return adj_attributes, name_attributes


def generate_any_description(action: str, attributes: Union[List[str],Dict[str,List[str]]]):
    env_params = get_env_params().copy()

    descriptions = []

    if action in ('Grasp', 'Grow', 'Attempt grow'):
        list_excluded = []
        if action == 'Grow':
            list_excluded = env_params['categories']['furniture'] + env_params['categories']['supply'] + ('furniture', 'supply')
        elif action == 'Attempted grow':
            list_excluded = env_params['categories']['living_thing'] + ('living_thing', 'animal', 'plant')

        if isinstance(attributes, dict):
            adjective_attributes, name_attributes = attributes['adjective_attributes'], attributes['name_attributes']
        else:
            adjective_attributes, name_attributes = sort_attributes(attributes)

        adjective_attributes = [a for a in adjective_attributes if a not in list_excluded]
        name_attributes = [a for a in name_attributes if a not in list_excluded]

        for adj in adjective_attributes:
            quantifier = 'any'
            for name in name_attributes:
                descriptions.append('{} {} {}'.format(action, adj, name))
            descriptions.append('{} {} {} thing'.format(action, quantifier, adj))
        for name in name_attributes:
            descriptions.append('{} any {}'.format(action, name))

        adj_combination = [c for c in env_params['combination_sentences'] if all(i in adjective_attributes for i in c.split())]

        for adj_comb in adj_combination:
            for name in name_attributes:
                if name not in list_excluded:
                    descriptions.append('{} {} {}'.format(action, adj_comb, name))
    elif action == 'Go':
        for pos in attributes:
            descriptions.append('Go {}'.format(pos))

    return descriptions.copy()

def generate_all_descriptions(env_params):
    """
    Generates all possible descriptions from a set of environment parameters.

    Parameters
    ----------
    env_params: dict
        Dict of environment parameters from get_env_params function.

    Returns
    -------
    training_descriptions: tuple of str
        Tuple of descriptions that belong to the training set (descriptions that do not contain occurrences reserved to the testing set).
    test_descriptions: tuple of str
        Tuple of descriptions that belong to the testing set (that contain occurrences reserved to the testing set).
    extra_descriptions: tuple of str
        Other descriptions that we might want to track (here when the agent tries to grow furniture for instance).

    """
    global descriptions_cache

    if descriptions_cache is not None:
        return descriptions_cache

    p = env_params.copy()

    # Get the list of admissible attributes and split them by name attributes (type and categories) and adjective attributes.
    name_attributes = env_params['name_attributes']
    adjective_attributes = env_params['adjective_attributes']
    
    attributes = {      
        "name_attributes": name_attributes + ('thing',),
        "adjective_attributes": adjective_attributes
    }


    # function to check whether an attribute is a relative attribute
    check_if_relative = env_params['extract_functions']['check_if_relative']
    # function to generate attributes as combinations of non-relative attributes.
    combine_two = env_params['extract_functions']['combine_two']


    # Add combined attributes if needed.
    if p['attribute_combinations']:
        adjective_attributes += combine_two(adjective_attributes, adjective_attributes)


    all_descriptions = ()
    
    if 'Move' in p['admissible_actions']:
        attrs = ['left', 'right', 'bottom', 'top', 'center']
        attrs += [f"{d2} {d1}" for d1 in ['left', 'right'] for d2 in ['top', 'bottom']] 
        
        move_descriptions = generate_any_description('Go', attrs)
        all_descriptions += tuple(move_descriptions)
    
    if 'Grasp' in p['admissible_actions']:
        grasp_descriptions = generate_any_description('Grasp', attributes)
        all_descriptions += tuple(grasp_descriptions)
    
    
    if 'Grow' in p['admissible_actions']:
        grow_descriptions = generate_any_description('Grow', attributes)
        all_descriptions += tuple(grow_descriptions)
        attempted_grow_descriptions = tuple(generate_any_description('Attempt grow', attributes))

    train_descriptions = []
    test_descriptions = []
    for descr in all_descriptions:
        to_remove = False
        for w in chain.from_iterable(p['words_test_set_def'].values()): # words_test_set_def is the set of occurrences that is reserved to the testing set.
            if isinstance(w, re.Pattern):
                r = w.search(descr)
                if r is not None:
                    to_remove = True
                    break
            elif w in descr:
                to_remove = True
                break
                
        if not to_remove:
            train_descriptions.append(descr)
        else:
            test_descriptions.append(descr)
    
    train_descriptions = tuple(sorted(train_descriptions))
    test_descriptions = tuple(sorted(test_descriptions))
    extra_descriptions = tuple(sorted(attempted_grow_descriptions))

    descriptions_cache = (train_descriptions, test_descriptions, extra_descriptions)
    return descriptions_cache

if __name__ == '__main__':
    env_params = get_env_params()
    train_descriptions, test_descriptions, extra_descriptions = generate_all_descriptions(env_params)
    