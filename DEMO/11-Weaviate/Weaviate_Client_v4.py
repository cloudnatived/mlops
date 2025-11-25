#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Weaviate_Client_v4.py
import weaviate
from weaviate.classes.init import Auth

# For basic connection (no authentication)
client = weaviate.connect_to_local(
    host="172.18.6.60",
    port=8080,
    grpc_port=50051  # if gRPC is available
)

# Or use the more explicit approach:
# client = weaviate.WeaviateClient(
#     connection_params=weaviate.ConnectionParams(
#         http_host="172.18.6.60",
#         http_port=8080,
#         http_secure=False,
#         grpc_host="172.18.6.60",
#         grpc_port=50051,
#         grpc_secure=False
#     )
# )

# Connect to the client
client.connect()

# Your operations here
print(client.is_ready())

# Don't forget to close the connection
client.close()
