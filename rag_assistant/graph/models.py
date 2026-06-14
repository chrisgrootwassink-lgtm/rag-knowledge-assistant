from pydantic import BaseModel


class Property(BaseModel):
    key: str
    value: str


class Node(BaseModel):
    node_label: str
    node_properties: list[Property]


class Relationship(BaseModel):
    relationship_type: str
    relationship_properties: list[Property]


class Edge(BaseModel):
    source: Node
    target: Node
    relationship: Relationship


class NodesResponse(BaseModel):
    nodes: list[Node]


class EdgesResponse(BaseModel):
    edges: list[Edge]


class GraphResponse(BaseModel):
    nodes: list[Node]
    edges: list[Edge]
