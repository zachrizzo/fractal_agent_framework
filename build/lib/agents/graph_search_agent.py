#fractal_framework/agents/graph_search_agent.py

from typing import Any, Dict, List
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
from .vector_search_agent import VectorSearchAgent
from core.fractal_task import FractalTask
from core.fractal_context import FractalContext
from core.fractal_result import FractalResult
from utils import logger


class GraphAwareVectorSearchAgent(VectorSearchAgent):
    """Enhanced vector search agent that considers graph structure"""

    def __init__(self, name: str):
        super().__init__(name)
        self.graph_embeddings = {}

    async def _process_task(self, task: FractalTask, context: FractalContext,
                          subtask_results: List[FractalResult]) -> FractalResult:
        operation = task.data.get('operation')

        if operation == 'index':
            return await self._handle_graph_indexing(context)
        elif operation == 'search':
            return await self._handle_graph_search(task, context)
        else:
            return await super()._process_task(task, context, subtask_results)

    async def _handle_graph_indexing(self, context: FractalContext) -> FractalResult:
        """Index code with graph structure awareness"""
        try:
            # Get subgraph embeddings for each node
            for node_id in context.graph.nodes():
                # Get node's code
                code = context.graph.nodes[node_id].get('code', '')
                if not code:
                    continue

                # Get local subgraph
                local_graph = self._extract_local_subgraph(context.graph, node_id)

                # Combine code and graph features
                combined_embedding = self._compute_combined_embedding(
                    code,
                    local_graph,
                    context.graph.nodes[node_id]
                )

                self.graph_embeddings[node_id] = combined_embedding

            return FractalResult("index", True, result={
                'num_indexed': len(self.graph_embeddings),
                'graph_features': self._analyze_graph_features(context.graph)
            })

        except Exception as e:
            logger.error(f"Error during graph indexing: {str(e)}")
            return FractalResult("index", False, error=str(e))

    def _extract_local_subgraph(self, graph: nx.DiGraph, node_id: str, depth: int = 2) -> nx.DiGraph:
        """Extract local subgraph around a node"""
        # Get nodes within k hops
        nodes = {node_id}
        current_depth = 0
        frontier = {node_id}

        while current_depth < depth and frontier:
            next_frontier = set()
            for node in frontier:
                # Add predecessors and successors
                next_frontier.update(graph.predecessors(node))
                next_frontier.update(graph.successors(node))
            nodes.update(next_frontier)
            frontier = next_frontier
            current_depth += 1

        return graph.subgraph(nodes)

    def _compute_combined_embedding(
        self,
        code: str,
        local_graph: nx.DiGraph,
        node_attrs: Dict[str, Any]
    ) -> np.ndarray:
        """Compute embedding that combines code and graph features"""
        # Get code embedding
        code_embedding = self.vectorizer.transform([code]).toarray()[0]

        # Get graph features
        graph_features = np.array([
            local_graph.number_of_nodes(),  # Size of local neighborhood
            local_graph.number_of_edges(),  # Local connectivity
            nx.density(local_graph),        # Local density
            len(list(nx.simple_cycles(local_graph))),  # Number of cycles
            self._get_node_centrality(local_graph, node_attrs)  # Node importance
        ])

        # Normalize graph features
        graph_features = (graph_features - graph_features.mean()) / (graph_features.std() + 1e-8)

        # Combine embeddings (with weights)
        combined = np.concatenate([
            code_embedding * 0.7,  # Code similarity is primary
            graph_features * 0.3   # Graph structure provides context
        ])

        return combined / np.linalg.norm(combined)

    def _get_node_centrality(self, graph: nx.DiGraph, node_attrs: Dict[str, Any]) -> float:
        """Calculate node importance based on multiple factors"""
        # Combine different centrality measures
        degree_cent = nx.degree_centrality(graph)
        between_cent = nx.betweenness_centrality(graph)
        close_cent = nx.closeness_centrality(graph)

        # Weight by node attributes
        type_weights = {
            'function': 1.0,
            'class': 1.2,
            'module': 1.5
        }
        type_weight = type_weights.get(node_attrs.get('type', ''), 1.0)

        # Combine metrics
        node_id = next(iter(graph.nodes()))  # Get the central node
        centrality = (
            degree_cent[node_id] * 0.3 +
            between_cent[node_id] * 0.3 +
            close_cent[node_id] * 0.4
        ) * type_weight

        return centrality

    async def _handle_graph_search(self, task: FractalTask, context: FractalContext) -> FractalResult:
        """Find similar code considering graph structure"""
        try:
            node_id = task.data.get('node_id')
            if not node_id or node_id not in context.graph:
                return FractalResult(task.id, False, error="Invalid node ID")

            # Get query embedding
            code = context.graph.nodes[node_id].get('code', '')
            local_graph = self._extract_local_subgraph(context.graph, node_id)
            query_embedding = self._compute_combined_embedding(
                code,
                local_graph,
                context.graph.nodes[node_id]
            )

            # Calculate similarities
            similarities = []
            for other_id, embedding in self.graph_embeddings.items():
                if other_id != node_id:
                    sim = {
                        'node_id': other_id,
                        'similarity': float(
                            cosine_similarity(
                                query_embedding.reshape(1, -1),
                                embedding.reshape(1, -1)
                            )[0, 0]
                        ),
                        'path_similarity': self._calculate_path_similarity(
                            context.graph, node_id, other_id
                        )
                    }
                    similarities.append(sim)

            # Sort by combined similarity
            similarities.sort(
                key=lambda x: x['similarity'] * 0.7 + x['path_similarity'] * 0.3,
                reverse=True
            )

            # Get top k results
            k = task.data.get('k', 5)
            top_results = similarities[:k]

            return FractalResult(task.id, True, result={
                'query_node': node_id,
                'similar_nodes': top_results,
                'graph_context': self._get_graph_context(context.graph, node_id, top_results)
            })

        except Exception as e:
            logger.error(f"Error during graph search: {str(e)}")
            return FractalResult(task.id, False, error=str(e))

    def _calculate_path_similarity(
        self,
        graph: nx.DiGraph,
        node1: str,
        node2: str
    ) -> float:
        """Calculate similarity based on graph paths"""
        try:
            # Get all simple paths between nodes
            paths = list(nx.all_simple_paths(graph, node1, node2, cutoff=4))
            if not paths:
                paths = list(nx.all_simple_paths(graph, node2, node1, cutoff=4))

            if not paths:
                return 0.0

            # Calculate path-based similarity
            min_path_length = min(len(path) for path in paths)
            num_paths = len(paths)

            # Combine factors (shorter paths and more paths indicate higher similarity)
            path_sim = 1.0 / (min_path_length + 1) * min(1.0, num_paths / 3)

            return path_sim

        except (nx.NetworkXNoPath, nx.NetworkXError):
            return 0.0

    def _get_graph_context(
        self,
        graph: nx.DiGraph,
        query_node: str,
        similar_nodes: List[Dict]
    ) -> Dict[str, Any]:
        """Get relevant graph context for the results"""
        context_nodes = {query_node}
        for node in similar_nodes:
            context_nodes.add(node['node_id'])

        # Get the subgraph connecting these nodes
        paths = []
        for node1, node2 in itertools.combinations(context_nodes, 2):
            try:
                path = nx.shortest_path(graph, node1, node2)
                paths.extend(path)
            except nx.NetworkXNoPath:
                continue

        context_graph = graph.subgraph(set(paths))

        return {
            'nodes': list(context_graph.nodes()),
            'edges': list(context_graph.edges()),
            'community_structure': self._analyze_community_structure(context_graph)
        }

    def _analyze_community_structure(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze the community structure of the context graph"""
        try:
            communities = nx.community.greedy_modularity_communities(graph.to_undirected())
            return {
                'num_communities': len(communities),
                'modularity': nx.community.modularity(graph.to_undirected(), communities),
                'community_sizes': [len(c) for c in communities]
            }
        except:
            return {}
