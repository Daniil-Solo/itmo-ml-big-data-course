from diagrams import Diagram
from diagrams.c4 import SystemBoundary, C4Node
from diagrams.onprem.client import User


def UseCase(name, **kwargs):
    use_case_attributes = {
        "name": name,
        "technology": None,
        "description": "",
        "type": "Use Case",
        "shape": "ellipse"
    }
    use_case_attributes.update(kwargs)
    return C4Node(**use_case_attributes)


with Diagram("[Use Case Diagram] Movie System", show=False, outformat="png"):
    customer = User("Customer")

    with SystemBoundary("Movie System"):
        action_get_recs = UseCase(name="Get recommendations")
        customer - action_get_recs
