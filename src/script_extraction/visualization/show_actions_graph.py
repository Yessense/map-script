from typing import List

from pyvis.network import Network

from src.script_extraction.text_preprocessing.extract_semantic_roles import Obj, Action


def add_image_node(net: Network, obj: Obj) -> None:
    for image in obj.images:
        net.add_node(image.index(), label=image.text, color="#FEF445", size=10)
        net.add_edge(obj.index(), image.index(), label=image.pos.value)


def add_action_node(net: Network, action) -> None:
    net.add_node(action.index(), label=action.text, color='#2D9BF0', size=17)
    for obj in action.cluster_objects:
        net.add_node(obj.index(), label=obj.text, color="#808080", size=14)
        net.add_edge(action.index(), obj.index(), label=obj.arg_type.value)
        add_image_node(net, obj)
    for act in action.actions:
        add_action_node(net, act)


def add_action_edge(net: Network, action: Action) -> None:
    for act in action.actions:
        net.add_edge(action.index(), act.index())
    for act in action.actions:
        add_action_edge(net, act)


def show_actions_graph(actions: List[Action]) -> None:
    """
    Just actions visualization
    Parameters
    ----------
    actions: List[Action]

    Returns
    -------

    """
    # create pyvis Network
    net = Network(notebook=True, height='100%', width='100%')

    net.add_node("0", label="Script", color='#F24726', size=20)
    for i, action in enumerate(actions):
        add_action_node(net, action)
        add_action_edge(net, action)
        net.add_edge("0", str(action.index()))
        if i:
            net.add_edge(str(actions[i - 1].index()), str(action.index()))

    net.set_options("""
    var options = {
      "physics": {
        "barnesHut": {
         "gravitationalConstant": -10050
          }
      }
    }
    """)
    net.show("Script.html")
