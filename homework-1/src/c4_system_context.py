from diagrams import Diagram
from diagrams.c4 import System, Relationship
from diagrams.onprem.client import User


graph_attr = {
    "splines": "spline",
}


with Diagram("[System Context Diagram] Movie System", show=False, outformat="png", graph_attr=graph_attr):
    customer = User("Customer")
    movie_system = System("Movie System", external=False, description="Provides all of the Movie System functionality")
    customer >> Relationship("gets personal recommendations") >> movie_system
