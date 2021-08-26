from typing import List, Union, Tuple, Dict, Any, Optional

from src.script_extraction.preprocessing.words_object import Action, WordsObject, POS, Obj, Position, Cluster


def resolve_phrases(items: List[Union[Action, Cluster]],
                    trees_list: List[List[Tuple[int, WordsObject]]],
                    text_info: Dict[str, Any]) -> List[Union[Action, Cluster]]:
    # finding child object which is phrase
    for item_index, item in enumerate(items):
        for i, obj in enumerate(item.objects):
            if obj.pos is POS.PHRASE:
                # candidates are inside phrase
                candidates_for_obj: List[Tuple[int, WordsObject]] = [
                    (level, words_obj) for level, words_obj in trees_list[obj.position.sentence_number]
                    if words_obj.position.inside(obj.position) and words_obj.is_accepted]

                test_candidates_for_obj = []
                for level, words_obj in trees_list[obj.position.sentence_number]:
                    inside = words_obj.position.inside(obj.position)
                    accepted = words_obj.is_accepted
                    passsing = inside and accepted
                    if passsing:
                        test_candidates_for_obj.append(words_obj)


                # removing phrase from .objects_signs
                item.objects[i] = None

                # if there are accepted candidates
                if len(candidates_for_obj):
                    min_level = min([level for level, image in candidates_for_obj])

                    new_objects: List[Union[WordsObject, Obj]] = select_new_objects(candidates_for_obj, min_level)
                    # to Obj class
                    new_objects = [Obj(text=words_object.text,
                                       # words spans from phrase for checking images inside
                                       # symbols spans from new word for finding this word in tree
                                       position=Position(sentence_number=obj.position.sentence_number,
                                                         start_word=obj.position.start_word,
                                                         end_word=obj.position.end_word,
                                                         start_symbol=words_object.position.start_symbol,
                                                         end_symbol=words_object.position.end_symbol),
                                       pos=words_object.pos,
                                       arg_type=obj.arg_type) for words_object in new_objects]
                    # add images
                    for phrase_shard in new_objects:
                        phrase_shard.images = find_object_images(obj, phrase_shard, text_info=text_info)
                    item.objects.extend(new_objects)
        # delete None objects_signs (phrases)
        item.objects = [obj for obj in item.objects if obj is not None]
    return items


def select_new_objects(candidates_for_obj: List[Tuple[int, WordsObject]],
                       min_level: int) -> List[WordsObject]:
    """
    Apply restrictions to candidates for action in phrase
    Usually its lowest level candidates
    Parameters
    ----------
    candidates_for_obj : List[Tuple[int, WOrdsObject]]
    min_level: int

    Returns
    -------
    out: List[WordsObject]

    """
    objects = [words_obj for level, words_obj in candidates_for_obj if level == min_level]
    return objects


def get_trees_list(text_info: Dict[str, Any]) -> List[List[Tuple[int, WordsObject]]]:
    """
    Convert tree to flat list with WordObject and his level(int) in tree

    Parameters
    ----------
    text_info

    Returns
    -------
    out : List[List[Tuple[int, WordsObject]]]
    """
    trees_list = []
    for i, sentence_info in enumerate(text_info['sentences_info']):
        trees_list.append(get_roots(sentence_info['dependency']['hierplane_tree']['root'],
                                    sentence_number=i))
        for level, words_object in trees_list[-1]:
            words_object.position.set_words_bounds(sentence_info['dependency']['words'])
    return trees_list


def get_roots(node: Dict[str, Any],
              sentence_number: int,
              words_list: Optional[List[WordsObject]] = None,
              level: Optional[int] = 0) -> List[Tuple[int, WordsObject]]:
    """
    Create list from one hierarchical tree

    Parameters
    ----------
    node: Dict[str, Any]
        node in hierpal tree in dependencies
    sentence_number: int
        sentence number, needed for Position
    words_list: Optional[List[WordsObject]]
        all founded words added here
    level:
        current iteration deep level

    Returns
    -------
    out : List[Tuple[int, WordsObject]
        deep level, WordsObject corresponding to word

    """
    if words_list is None:
        words_list = []
    words_list.append((level,
                       WordsObject(text=node['word'].lower(),
                                   pos=POS(node['attributes'][0]),
                                   position=Position(start_symbol=node['spans'][0]['start'],
                                                     end_symbol=node['spans'][0]['end'],
                                                     sentence_number=sentence_number))))
    if 'children' in node:
        for child in node['children']:
            get_roots(child, sentence_number, words_list, level + 1)
    return words_list


def find_object(obj: Obj,  # part-of-phrase object
                node: Dict[str, Any],
                objects: Optional[List[Dict[str, any]]] = None) -> List[Dict[str, Any]]:
    """
    Recursive function to find node at same position as `action`
    Parameters
    ----------
    obj: Obj
        Main part of phrase
    node: Dict[str, Any]
        Root node in sentence
    objects: Optional[List[Dict[str, Any]]] = None
        for returning one founded item

    Returns
    -------
    out: List[Dict[str, Any]]

    """
    if objects is None:
        objects = []
    spans: Dict[str, int] = next(iter(node['spans']))
    if obj.position.start_symbol == spans['start'] and obj.position.end_symbol == spans['end']:
        objects.append(node)
    else:
        if 'children' in node:
            for child in node['children']:
                find_object(obj, child, objects)
    return objects


def find_object_images(parent_obj: Union[WordsObject, Action],
                       obj: Obj,  # part-of-phrase object
                       text_info: Dict[str, Any]) -> List[WordsObject]:
    object_images = []

    # find corresponding node to given object
    # recursive function returns list with 1 object
    node = find_object(obj, text_info['sentences_info'][
        obj.position.sentence_number]['dependency']['hierplane_tree']['root'])

    # check for error, but it shouldn't happen
    if not len(node):
        return object_images
    else:
        node = node[0]

    # find images
    if 'children' in node:
        for child in node['children']:
            spans: Dict[str, int] = next(iter(child['spans']))
            words_obj = WordsObject(text=child['word'].lower(),
                                    position=Position(sentence_number=obj.position.sentence_number,
                                                      start_symbol=spans['start'],
                                                      end_symbol=spans['end']),
                                    pos=POS(child['attributes'][0]))
            if words_obj.position.inside(parent_obj.position) and words_obj.may_be_image:
                object_images.append(words_obj)
    return object_images
