import os
import pickle
import re
import uuid
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from threading import RLock

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from networkx import Graph
from rich.markdown import Markdown
from rich.rule import Rule

from goose.notifier import Notifier
from goose.toolkit.base import Requirements, Toolkit, tool
from goose.toolkit.utils import RULEPREFIX, RULESTYLE


class NodeNotFoundError(Exception):
    pass


class NetworkBuilder(Toolkit):
    def __init__(self, notifier: Notifier, requires: Requirements) -> None:
        super().__init__(notifier=notifier, requires=requires)
        self.graph = Graph()
        self.local_graph_dir = Path(".goose/network")
        self.global_graph_dir = Path.home() / ".config/goose/network"
        self.title_index = {} # Maps titles to a single node ID

        self.tag_set = set()  # Centralized set of tags
        self.tag_index = defaultdict(set)  # Maps tags to node IDs
        self.lock = RLock() 
        self._ensure_graph_dirs()
    
    def _ensure_graph_dirs(self) -> None:
        """Ensure node directories exist"""
        self.local_graph_dir.parent.mkdir(parents=True, exist_ok=True)
        self.local_graph_dir.mkdir(exist_ok=True)
        self.global_graph_dir.parent.mkdir(parents=True, exist_ok=True)
        self.global_graph_dir.mkdir(exist_ok=True)

    def _sanitize_input(self, input_value: any, max_length: int = 255) -> str:
        """
        Sanitize input strings to prevent injection attacks and enforce length constraints.

        Parameters:
            input_value (any): The input to sanitize.
            max_length (int, optional): Maximum allowed length of the string.

        Returns:
            input (str): The sanitized string.
        """

        if not isinstance(input_value, str):
            input_value = str(input_value)
        if max_length is not None and len(input_value) > max_length:
            raise ValueError(
                f"Input exceeds maximum length of {max_length}."
                 "Try splitting the content into multiple entities"
            )
        # Perform basic sanitization
        sanitized = input_value.strip()
        if not sanitized:
            raise ValueError("Input cannot be empty or whitespace.")
        return sanitized
    
    def _sanitize_tag(self, tag: str) -> str:
        """Sanitize and standardize tag strings"""
        sanitized_tag = self._sanitize_input(tag, max_length=50).lower()
        sanitized_tag = sanitized_tag.replace(' ', '_')
        return sanitized_tag

    def _validate_dict_keys(self, item: dict, required_keys: list[str]):
        """Validate all required keys exist in a dictionary"""
        for arg in required_keys:
            if arg not in item:
                return False    
        return True
    
    def system(self) -> str:
        return f"""Use this toolkit to create strongly linked relationships.
        Learn from past experiences to solve previously unseen problems, faster.
        When encountering a task, search for related tags to understand what's been done in the past.
        """
    
    @tool
    def add_node(
        self,
        title: str,
        summary: str,
        tags: list[str] = [],
        **metadata: any
    ) -> str:
        """
        Add a node with metadata and tags to the graph

        Parameters:
            title (str): Title of the node.
            summary (str): Summary or content of the node.
            tags (list, optional): List of tags to associate with the node.
            metadata (any): Additional metadata as key-value pairs.

        Returns:
            node_id (str): The unique ID of the added node.

        Raises:
            ValueError: If title or summary is invalid.
        """
        with self.lock:
            # Validate and sanitize inputs
            title = self._sanitize_input(title, max_length=255)
            summary = self._sanitize_input(summary, max_length=1024)
            sanitized_metadata = {k: self._sanitize_input(v) for k, v in metadata.items()}

            # Validate tags
            node_tags = set()
            for tag in tags:
                sanitized_tag = self._sanitize_tag(tag)
                self.tag_set.add(sanitized_tag)
                node_tags.add(sanitized_tag)

            if title in self.title_index:
                raise ValueError(f"Duplicate title: {self.title_index[title]} has the title '{title}'")
            
            node_id = str(uuid.uuid4())[:8]
            self.graph.add_node(node_id, title=title, summary=summary, tags=list(node_tags), **sanitized_metadata)

            # Update indexes
            self.title_index[title] = node_id
            for tag in node_tags:
                self.tag_index[tag].add(node_id)


            node_tags = f"\n[{', '.join(sorted(tags))}]" if tags else ""
            node_metadata = f"\n**Metadata:** {metadata}" if metadata else ""
            node_details = f"\n - **{title}** {node_tags}{node_metadata}\n"
            self.notifier.log(Markdown(node_details))
            return node_id

    @tool
    def add_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: str,
        **metadata: any
    ) -> None:
        """
        Add an edge between two nodes

        Parameters:
            from_node_id (str): The ID of the source node.
            to_node_id (str): The ID of the target node.
            relationship_type (str): The type of relationship.
            metadata (any): Additional metadata for the relationship.

        Raises:
            NodeNotFoundError: If either node ID does not exist.
            ValueError: If attempting to create a relationship with invalid parameters.
        """
        with self.lock:
            if from_node_id not in self.graph:
                raise NodeNotFoundError(f"Node with ID '{from_node_id}' not found.")
            if to_node_id not in self.graph:
                raise NodeNotFoundError(f"Node with ID '{to_node_id}' not found.")
            if from_node_id == to_node_id:
                raise ValueError("Cannot create a relationship from a node to itself.")
            
            self.graph.add_edge(
                from_node_id,
                to_node_id,
                relationship=relationship_type,
                **metadata
            )
            from_title = self.get_node(from_node_id)['title']
            to_title = self.get_node(to_node_id)['title']

            self.notifier.log(f"- '{relationship_type}' added from '{from_title}' to '{to_title}'.")
    
    @tool
    def get_relationships(self, node_id: str) -> list:
        """
        Retrieve all relationships associated with a specific node in the network.

        Parameters:
            node_id (str): The ID of the node.

        Returns:
            relationships (list[dict]): A list of relationship details.

        Raises:
            NodeNotFoundError: If the node ID does not exist.
        """
        self.notifier.status('Getting relationships')
        with self.lock:
            if node_id not in self.graph:
                raise NodeNotFoundError(f"Node with ID '{node_id}' not found.")
            relationships = []
            for neighbor in self.graph.neighbors(node_id):
                edge_data = self.graph.get_edge_data(node_id, neighbor)
                relationship = {
                    'to_node_id': neighbor,
                    'relationship_type': edge_data.get('relationship_type', 'relates_to'),
                    'metadata': edge_data.get('metadata', {})
                }
                relationships.append(relationship)
            return relationships
    
    @tool
    def get_neighbors(
        self,
        node_id: str,
        depth: int = 3,
        title_only: bool = False,
    ) -> dict:
        """
        View neighbors connected to a specific node, up to a certain depth

        Parameters:
            node_id (str): The ID of the starting node.
            depth (int): The depth of traversal. Defaults to 3.
            title_only (bool): If True, return only titles. Defaults to False.
        Returns:
            nodes (dict): A dictionary of connected nodes up to the specified depth.
        """
        self.notifier.log('Getting neighbors for node')
        if not isinstance(depth, int) or depth < 1:
            raise ValueError("Depth must be a positive integer.")

        with self.lock:
            if node_id not in self.graph:
                raise NodeNotFoundError(f"Node with ID '{node_id}' not found.")

            visited = set()
            result = {}
            queue = deque([(node_id, 0)])
            visited.add(node_id)

            while queue:
                current_node, current_depth = queue.popleft()
                if current_depth > 0 and current_depth <= depth:
                    if title_only:
                        result[current_node] = self.graph.nodes[current_node].get('title', '')
                    else:
                        result[current_node] = self.graph.nodes[current_node]

                if current_depth < depth:
                    for neighbor in self.graph.neighbors(current_node):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, current_depth + 1))

            return result

    @tool
    def add_nodes_with_relationships(self, nodes: list[dict], relationships: list[dict]) -> dict[str, str]:
        """
        Adds nodes with associated relationships to the Graph in a single operation

        Args:
            nodes (list[dict]): A list of nodes
                - title (str): Title of the node.
                - summary (str): Summary of the node.
                - tags (list[str]): Tags associated with the node.

            relationships (list[dict]): A list of relationships with keys
                - from_node_title (str): Title of the source node.
                - to_node_title (str): Title of the target node.
                - relationship_type (str): Type of the relationship.

        Returns:
            node_id_map (dict[str, str]): A mapping of node titles to their unique IDs.
        """

        self.notifier.log(Rule(RULEPREFIX + f"Connecting {len(nodes)} nodes to the graph", style=RULESTYLE, align="left"))

        required_relationship_args = ['from_node_title', 'to_node_title', 'relationship_type']
        for relationship in relationships:
            if not self._validate_dict_keys(relationship, required_relationship_args):
                raise ValueError(f"Relationships require keys: {required_relationship_args}")
        required_node_args = ["title", "summary", "tags"]
        for node in nodes:
            if not self._validate_dict_keys(node, required_node_args):
                raise ValueError(f"Nodes require keys: {required_node_args}")
            
            
        node_id_map = {}  # Maps node titles to their unique IDs

        # Check for duplicates across the entire graph
        for node in nodes:
            if node['title'] in self.title_index and self.title_index[node['title']]:
                raise ValueError(f"Node title '{node['title']}' already exists in the graph.")

        with self.lock:
            for node in nodes:
                # Add the node
                node_id = self.add_node(
                    title=node['title'],
                    summary=node['summary'],
                    tags=node['tags'], 
                )
                node_id_map[node['title']] = node_id

            for relationship in relationships:
                from_title = relationship['from_node_title']
                to_title = relationship['to_node_title']
                rel_type = relationship['relationship_type']

                # Retrieve node IDs from titles
                from_id = node_id_map.get(from_title)
                to_id = node_id_map.get(to_title)

                if not from_id:
                    raise ValueError(f"Node with title '{from_title}' not found in provided nodes.")
                if not to_id:
                    raise ValueError(f"Node with title '{to_title}' not found in provided nodes.")

                self.add_relationship(
                    from_node_id=from_id,
                    to_node_id=to_id,
                    relationship_type=rel_type,
                )

        return node_id_map
    
    def add_tag(self, tag: str) -> None:
        """
        Add a new tag to the tag set.

        Parameters:
            tag (str): The tag to be added.

        """
        with self.lock:
            if tag in self.tag_set:
                return "Tag added"
            self.tag_set.add(tag)
        
        return f"Added tag: {tag}"


    ### Node Management Methods ###
    @tool
    def update_node(
        self,
        node_id: str,
        title: str = '',
        summary: str = '',
        tags: list[str] = [],
        **metadata: any
    ) -> None:
        """
        Update an existing node's details and tags.

        Parameters:
            node_id (str): The ID of the node to update.
            title (str, optional): New title.
            summary (str, optional): New summary.
            tags (list, optional): New list of tags.
            metadata (any): Additional metadata to update.

        Raises:
            NodeNotFoundError: If the node ID does not exist.
        """
        with self.lock:
            if node_id not in self.graph:
                raise NodeNotFoundError(f"Node with ID '{node_id}' not found.")

            node = self.graph.nodes[node_id]

            # Update title
            if title:
                sanitized_title = self._sanitize_input(title, max_length=255)
                old_title = node['title']
                node['title'] = sanitized_title
                # Update title index
                self.title_index[old_title].remove(node_id)
                if not self.title_index[old_title]:
                    del self.title_index[old_title]
                self.title_index[sanitized_title].append(node_id)
                self.notifier.log(f"Node ID={node_id} title updated from '{old_title}' to '{sanitized_title}'")

            # Update summary
            if summary:
                node['summary'] = self._sanitize_input(summary, max_length=1024)
                self.notifier.log(f"Node ID={node_id} summary updated.")

            # Update metadata
            reserved_keys = {'title', 'summary', 'tags'}
            for key, value in metadata.items():
                if key in reserved_keys:
                    raise ValueError(f"Cannot overwrite reserved key '{key}' in metadata.")
                node[key] = self._sanitize_input(value)

            # Update tags
            old_tags = set(node.get('tags', []))
            for tag in old_tags:
                self.tag_index[tag].discard(node_id)
                if not self.tag_index[tag]:
                    del self.tag_index[tag]

            # Validate and sanitize new tags
            new_tags = set()
            for tag in tags:
                sanitized_tag = self._sanitize_tag(tag)
                if sanitized_tag not in self.tag_set:
                    self.tag_set.add(tag)
                new_tags.add(sanitized_tag)

            # Update node tags
            node['tags'] = list(new_tags)

            # Update tag index
            for tag in new_tags:
                self.tag_index[tag].add(node_id)

            self.notifier.log(f"Node ID={node_id} tags updated to {list(new_tags)}")

    @tool
    def get_node(self, node_id: str) -> dict[str, str]:
        """
        Retrieve a node and its metadata by ID.

        Parameters:
            node_id (str): The ID of the node to retrieve.

        Returns:
           node_data (dict[str, str]): The node's data.

        Raises:
            NodeNotFoundError: If the node ID does not exist.
        """
        with self.lock:
            if node_id not in self.graph:
                raise NodeNotFoundError(f"Node with ID '{node_id}' not found.")
            return self.graph.nodes[node_id]


    @tool
    def get_node_by_title(self, title: str) -> dict:
        """
        Search and retrieve a node by title.

        Parameters:
            title (str): The title to search for.

        Returns:
            node (dict): The matching node
        """
        with self.lock:
            sanitized_title = self._sanitize_input(title, max_length=255)
            if sanitized_title not in self.title_index:
                return NodeNotFoundError(f"Node {sanitized_title} does not exist")
            node_id = self.title_index[sanitized_title]
            return self.get_node(node_id)

    @tool
    def search_nodes_by_title(self, search_term: str) -> list:
        """
        Search and retrieve nodes by partial title match.

        Parameters:
            search_term (str): The substring to search for in titles.

        Returns:
            nodes (list): A list of tuples containing node IDs and their data.
        """
        self.notifier.status(f'searching by {search_term}')
        with self.lock:
            pattern = re.compile(re.escape(search_term), re.IGNORECASE)
            matching_titles = filter(pattern.search, self.title_index.keys())
            result = []
            for title in matching_titles:
                for node_id in self.title_index[title]:
                    result.append((node_id, self.graph.nodes[node_id]))
            return result

    @tool
    def search_nodes_by_tags(
        self,
        tags: list[str],
        match_all: bool = False
    ) -> list[tuple[str, dict[str, str]]]:
        """
        Search and retrieve nodes by tags.

        Parameters:
            tags (list): List of tags to search for.
            match_all (bool): If True, only return nodes that have all the specified tags.
                              If False, return nodes that have any of the specified tags.

        Returns:
            nodes (list[tuple[str, dict]]): A list of tuples containing node IDs and their data.
        """
        with self.lock:
            sanitized_tags = [self._sanitize_tag(tag) for tag in tags]
            for tag in sanitized_tags:
                if tag not in self.tag_set:
                    return f"Tag '{tag}' does not exist. Use List tags"

            if not sanitized_tags:
                return []

            # Retrieve sets of node IDs for each tag
            node_sets = [self.tag_index[tag] for tag in sanitized_tags]

            # Compute intersection or union based on match_all
            if match_all:
                matched_nodes = set.intersection(*node_sets) if node_sets else set()
            else:
                matched_nodes = set.union(*node_sets) if node_sets else set()

            return [(node_id, self.graph.nodes[node_id]) for node_id in matched_nodes]

    @tool
    def add_lazy_relationship(self, from_node: dict, to_node: dict, relationship_type: str, **metadata):
        """
        Adds a from_node and to_node to the graph with the specified relationship

        Parameters:
          from_node (dict): dictionary with keys title, summary, tags (list)
          to_node (dict): dictionary with keys tite, summary, tags (list)
          relationship_type (str): Specific relationship relating two nodes
          metadata (any): Additional {key: value} data to add to the relationship
        """
        from_node_id = self.add_node(from_node["title"], from_node["summary"], from_node["tags"])
        to_node_id = self.add_node(to_node["title"], to_node["summary"], to_node["tags"])
        print("LAZY RELATIONSHIP")
        self.add_relationship(from_node_id, to_node_id, relationship_type, **metadata)      

        return f"Created {from_node_id} ({from_node["title"]}) "\
        f"and {to_node_id} ({to_node['title']}) "\
        f"with a {relationship_type.upper()} edge"

    @tool
    def remove_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: str = ''
    ) -> None:
        """
        Remove a specific relationship between two nodes.

        Parameters:
            from_node_id (str): The ID of the source node.
            to_node_id (str): The ID of the target node.
            relationship_type (str, optional): The type of relationship to remove. If None, removes all relationships between the two nodes.

        Raises:
            NodeNotFoundError: If either node ID does not exist.
            ValueError: If the specified relationship does not exist.
        """
        with self.lock:
            if from_node_id not in self.graph:
                raise NodeNotFoundError(f"Node with ID '{from_node_id}' not found.")
            if to_node_id not in self.graph:
                raise NodeNotFoundError(f"Node with ID '{to_node_id}' not found.")
            if not self.graph.has_edge(from_node_id, to_node_id):
                raise ValueError(f"No relationship exists between '{from_node_id}' and '{to_node_id}'.")
            
            # If a specific relationship type is provided, check and remove it
            if relationship_type:
                edge_data = self.graph.get_edge_data(from_node_id, to_node_id)
                if edge_data.get('relationship_type') != relationship_type:
                    raise ValueError(f"Relationship type '{relationship_type}' does not match the existing relationship.")
                self.graph.remove_edge(from_node_id, to_node_id)
            else:
                self.graph.remove_edge(from_node_id, to_node_id)


    @tool
    def save_graph(self, filename: str="node_graph.pkl") -> str:
        """
        Saves the graph to a global graph directory

        Parameters:
            filename (str): The name of the output file with a .pkl extension

        Returns:
            message (str): Success message

        Raises:
            IOError: If the file cannot be written.
        """
        try:
            save_dir = self.global_graph_dir / filename
            with open(save_dir, 'wb') as file:
                pickle.dump({
                    'graph': self.graph,
                    'title_index': self.title_index,
                    'tag_set': self.tag_set,
                    'tag_index': self.tag_index,
                }, file)

            self.notifier.log(f"Graph successfully saved to {filename}.")
            return f"Graph saved to {filename}."
        except IOError as e:
            raise IOError(f"Failed to save graph: {e}")

    @tool
    def load_graph(self, filename) -> str:
        """
        Load the graph from a global graph directory

        Parameters:
            filename (str): A valid file with a .pkl extension

        Raises:
            IOError: If the file cannot be read.
        """
        try:
            filename = self.global_graph_dir / filename
            with open(filename, 'rb') as file:
                data = pickle.load(file)
                self.graph = data['graph']
                self.title_index = data['title_index']
                self.tag_set = data['tag_set']
                self.tag_index = data['tag_index']

                return "Graph loaded succesfully"
        except (IOError, pickle.UnpicklingError) as e:
            raise IOError(f"Failed to load graph: {e}")
    
    @tool
    def reset_network(self, create_backup: bool = False) -> None:
        """
        Resets the network

        Parameters:
           create_backup (bool): Whether to save the graph before resetting
        """
        # TODO: can we prompt user for confirmation?
        with self.lock:
            if create_backup:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"graph_backup_{timestamp}.png"
                self.save_graph(filename)
            self.graph = Graph()
            self.title_index = {}
            self.tag_set = set()
            self.tag_index = defaultdict(set) 
                  
    @tool
    def create_graphic(
        self,
        node_size: int = 500,
        node_color: str = 'skyblue',
        tagged_node_colors: dict[str, str] = None,
        edge_color: str = 'blue',
        figsize: tuple[int, int] = (12, 8),
        with_labels: bool = True,
        with_edge_labels: bool = True,
        font_size: int = 10,
        title: str = "Node Graph",
        save_dir: str = "graph_images",
        filename: str = '',
    ) -> str:
        """
        Save an image of the graph using matplotlib with aesthetic enhancements
    
        Parameters:
            node_size (int): Size of the nodes.
            node_color (str): Color for node nodes.
            tagged_node_colors (dict[str, str]): Tag names and their associated node colors.
            edge_color (str): Color for relationships.
            figsize (tuple[int, int]): Size of the matplotlib figure.
            with_labels (bool): Whether to display labels on nodes.
            with_edge_labels (bool): Whether to display labels on edges
            font_size (int): Font size for labels.
            title (str): Title of the graph.
            save_dir (str): Directory to save the graph image.
            filename (str): Specific filename for the saved image.
        Returns:
            Result (str): A string in the format "image:<path_to_image>".
        """
        plt.figure(figsize=figsize)
        G = self.graph.copy()
        if len(G.nodes) == 0:
            return "Nothing to create. No nodes have been added to the graph"
        
        node_colors = []
        node_shapes = []
        for node_id, data in G.nodes(data=True):
            if tagged_node_colors:
                matched_tag = None
                for tag, color in tagged_node_colors.items():
                    if tag in data.get('tags', []):  # Check if the node has the tag
                        matched_tag = color
                        break
                node_colors.append(matched_tag if matched_tag else 'grey')  # Default to grey if no tag matched
                node_shapes.append('^')  # Triangle for tagged nodes
            if 'title' in data:
                node_colors.append(node_color)
                node_shapes.append('o')  # Circle for nodes
            else:
                node_colors.append('grey')  # Default color
                node_shapes.append('^')  # Triangle for undefined types
    
        # Define edge colors based on relationship type
        edge_colors = []
        edge_relationship_types = set()
        edge_relationship_types = set() 

        for u, v, data in G.edges(data=True):
            rel_type = data.get('relationship_type', 'relates_to')
            edge_colors.append(edge_color)
            edge_relationship_types.add(rel_type)

        # Choose a layout for the graph
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
        # Draw nodes with different shapes
        unique_shapes = set(node_shapes)
        for shape in unique_shapes:
            shaped_nodes = [node for node, shp in zip(G.nodes(), node_shapes) if shp == shape]
            shaped_node_colors = [node_colors[i] for i, shp in enumerate(node_shapes) if shp == shape]
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=shaped_nodes,
                node_color=shaped_node_colors,
                node_size=node_size,
                node_shape=shape,
                alpha=0.9
            )
    
        # Draw edges
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=edge_colors,
            width=2,
            alpha=0.7,
            arrows=True,
            arrowstyle='->',
            arrowsize=15
        )

        # Add edge relationship types to the legend if edge labels are not shown
        legend_handles = []
        if with_edge_labels:
            for rel_type in edge_relationship_types:
                legend_handles.append(mlines.Line2D([], [], color=edge_colors[0], label=rel_type, linewidth=2))

        if with_edge_labels:
            # Draw edge labels
            edge_labels = {}
            for u, v, data in G.edges(data=True):
                if 'relationship_type' in data:
                    edge_labels[(u, v)] = data['relationship_type']

            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                font_size=font_size,
                font_color='red'
            )
            
        # Draw labels
        if with_labels:
            node_labels = {}
            for node_id, data in G.nodes(data=True):
                if 'title' in data:
                    node_labels[node_id] = data['title']
                else:
                    node_labels[node_id] = node_id  # Fallback to node ID
            
            nx.draw_networkx_labels(
                G,
                pos,
                labels=node_labels,
                font_size=font_size,
                font_color='black'
            )

        # Define legend elements
        node_patch = mpatches.Patch(color=node_color, label='Node')
        legend_handles.append(node_patch)
        
        if tagged_node_colors:
            for tag, color in tagged_node_colors.items():
                legend_handles.append(mpatches.Patch(color=color, label=tag))
        
    
        # Position legend outside the plot
        plt.legend(
            handles=legend_handles,
            loc='upper left',
            bbox_to_anchor=(1, 1),
            fontsize='small'
        )
    
        plt.title(title)
        plt.axis('off') 
        plt.tight_layout()
        
        image_dir = self.local_graph_dir / save_dir
        image_dir.mkdir(parents=True, exist_ok=True)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"node_graph_{timestamp}.png"
        else:
            if not filename.endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
                filename += '.png' 

        save_path = image_dir / filename

        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
            
        if os.path.exists(save_path):
            return f"image:{save_path}"
        
        return "ERROR: Could not save graph"

    @tool
    def print_graph(self) -> str:
        """
        Print the entire graph structure, including nodes and edges with their attributes.
        Parameters:
           None

        Returns:
            graph (str): A textual representation of the graph
        """
        text_representation = "\n--- Graph Nodes ---"
        with self.lock:
            print(text_representation)
            for node_id, data in self.graph.nodes(data=True):
                text_representation += f"{node_id}: {data['title']}"
                print(f"{node_id}: {data['title']}")
            print("\n--- Graph Edges ---")
            for u, v, data in self.graph.edges(data=True):
                text_representation += f"{u} --{data}--> {v}"
                print(f"{u} --{data}--> {v}")
        
        return text_representation
