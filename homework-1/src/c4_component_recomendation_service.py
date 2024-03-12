from diagrams import Diagram
from diagrams.c4 import Container, C4Node, SystemBoundary, Relationship


def Component(name, technology="", description="", **kwargs):
    use_case_attributes = {
        "name": name,
        "technology": technology,
        "description": description,
        "type": "Component",
        "shape": "rectangle"
    }
    use_case_attributes.update(kwargs)
    return C4Node(**use_case_attributes)


graph_attr = {
    "splines": "spline",
}


with Diagram("[Component Diagram] Recommendation Service", show=False, outformat="png", graph_attr=graph_attr):
    api = Container(
        name="API Gateway",
        technology="Uvicorn+FastAPI",
        description="Provides all of the Movie System functionality to users via JSON/HTTP API"
    )
    s3 = Container(name="S3", technology="MinIO", description="Stores the model for recommendations")

    with SystemBoundary("Recommendation Service"):
        grpc_server = Component(name="gRPC Server", technology="gRPC Python Server", description="Handles gRPC calls")
        model_loader = Component(
            name="Model Loader",
            technology="Python",
            description="Loads model and other artifacts for recommendations"
        )
        model = Component(name="Model", technology="Python", description="Generates recommendations for user")

    api >> Relationship("makes gRPC calls to [gRPC]") >> grpc_server
    model_loader >> Relationship("loads model from") >> s3
    model_loader >> Relationship("prepares") >> model
    grpc_server >> Relationship("uses") >> model
