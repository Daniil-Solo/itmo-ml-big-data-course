from diagrams import Diagram
from diagrams.c4 import Container, Relationship, Cluster, Person


graph_attr = {
    "splines": "spline",
}


with Diagram("[Container Diagram] Movie System", show=False, outformat="png", graph_attr=graph_attr):
    customer = Person("Customer")
    with Cluster("Movie System"):
        spa = Container(
            name="Single-Page Application", technology="JavaScript and React",
            description="Provides all of the Movie System functionality to users via their web browser"
        )
        nginx = Container(name="Web Server", technology="Nginx", description="Delivers the static content and SPA")
        api = Container(
            name="API Gateway", technology="Uvicorn+FastAPI",
            description="Provides all of the Movie System functionality to users via JSON/HTTP API"
        )
        recommender = Container(
            name="Recommendation Service", technology="gRPC Python Server",
            description="Provides recommendations for the customer via gRPC"
        )
        s3 = Container(name="S3", technology="MinIO", description="Stores the model for recommendations")
        kafka = Container(
            name="Interaction Event Bus", technology="Kafka Topic",
            description="Stores the events about user interactions with movies"
        )
        interaction_loader = Container(
            name="Interaction Loader", technology="Spark Streaming",
            description="Transfers data from the event bus to the storage"
        )
        hdfs = Container(
            name="Interaction Storage", technology="HDFS",
            description="Stores the interactions of customers with movies"
        )
        trainer = Container(
            name="Model Trainer", technology="Spark", description="Trains the model on a schedule every 2 days"
        )

    customer >> Relationship("gets personal recommendations using [HTTP]") >> spa
    nginx >> Relationship("delivers static content to the customer's web browser [HTTP]") >> spa
    spa >> Relationship("makes API calls to [HTTP]") >> api
    api >> Relationship("makes gRPC calls to [gRPC]") >> recommender
    api >> Relationship("sends interaction events using") >> kafka
    recommender >> Relationship("loads model from") >> s3
    interaction_loader >> Relationship("pulls events from") >> kafka
    interaction_loader >> Relationship("saves interactions into") >> hdfs
    trainer >> Relationship("loads interactions from") >> hdfs
    trainer >> Relationship("saves the trained model into") >> s3
