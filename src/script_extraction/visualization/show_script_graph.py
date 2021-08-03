from typing import Dict

from mapcore.swm.src.components.semnet import Sign, Connector
from pyvis.network import Network


def add_nodes(net: Network,
              signs_dict: Dict[str, Sign],
              color: str,
              size: int,
              add_signifs: bool = False,
              add_value: int = 0) -> None:
    for s in signs_dict.values():
        net.add_node(str(s.__hash__() + add_value), label=s.name, color=color, size=size)
        if add_signifs:
            for cm in s.significances.values():
                net.add_node(str(hash(cm)), label=f'{s.name}:{cm.index}', color=color, size=size - 5)
                net.add_edge(str(hash(s) + add_value), str(hash(cm)))


def add_image_edges(net: Network, obj: Sign) -> None:
    for cause in obj.meanings[1].cause:
        image: Sign = next(iter(cause.coincidences)).out_sign
        net.add_edge(str(hash(obj) + 2), str(hash(image) + 3))


def add_object_edges(net: Network, action: Sign) -> None:
    for cm in action.significances.values():
        for cause in cm.cause:
            connector: Connector = next(iter(cause.coincidences))
            out_index: int = connector.out_index
            role_sign: Sign = connector.out_sign
            label: str = role_sign.name
            obj_sign: Sign = next(iter(role_sign.significances[out_index].cause[0].coincidences)).out_sign
            net.add_edge(str(hash(action) + 1), str(hash(obj_sign) + 2), label=label)


def add_script_edges(net: Network,
                     S: Sign) -> None:
    for cause in S.significances[1].cause:
        connector: Connector = next(iter(cause.coincidences))
        to: Sign = connector.out_sign
        label = cause.order

        net.add_edge(str(S.__hash__()), str(to.__hash__() + 1), label=label)


def show_script_graph(S: Sign,
                      actions_signs: Dict[str, Sign],
                      objects_signs: Dict[str, Sign],
                      roles_signs: Dict[str, Sign],
                      images_signs: Dict[str, Sign]) -> None:
    net = Network(height='100%', width='100%')

    # net.add_node(str(S.__hash__()), label=S.name, color='#F24726', size=20)
    add_nodes(net, actions_signs, color='#2D9BF0', size=17, add_value=1)
    add_nodes(net, objects_signs, color="#808080", size=14, add_value=2)
    add_nodes(net, images_signs, color="#FEF445", size=10, add_value=3)

    # add_script_edges(net, S)

    for action in actions_signs.values():
        add_object_edges(net, action)
    for obj in objects_signs.values():
        add_image_edges(net, obj)

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