import json
import logging
import os
import pickle
import re
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Dict, List, Tuple, Union
from uuid import uuid4

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from networkx import Graph
from rich.markdown import Markdown
from rich.rule import Rule

from goose.notifier import Notifier
from goose.toolkit.utils import RULEPREFIX, RULESTYLE

# Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


class MemoryNotFoundError(Exception):
    pass


class TagError(Exception):
    def __init__(self, message: str, associated_memories: list[dict[str, any]] = []):
        super().__init__(message)
        self.associated_memories = associated_memories or []


from pathlib import Path

from goose.toolkit.base import Requirements, Toolkit, tool


class MemoryGraph(Toolkit):
    def __init__(self, notifier: Notifier, requires: Requirements) -> None:
        super().__init__(notifier=notifier, requires=requires)
        self.graph = Graph()
        self.local_graph_dir = Path(".goose/graph")
        self.global_graph_dir = Path.home() / ".config/goose/graph"
        self.title_index = defaultdict(list)
        self.tag_set = set()  # Centralized set of tags
        self.tag_index = defaultdict(set)  # Maps tags to memory IDs
        self.audit_log = []  # For audit logging
        self.lock = RLock() 
        
        self._ensure_graph_dirs()
   
    @tool
    def show_graph(
        self,
        node_size: int = 500,
        node_color_memories: str = 'skyblue',
        node_color_notes: str = 'lightgreen',
        tagged_node_colors: dict[str, str] = None,
        edge_color: str = 'blue',
        edge_color_has_note: str = 'green',
        figsize: tuple[int, int] = (12, 8),
        with_labels: bool = True,
        with_edge_labels: bool = False,
        font_size: int = 10,
        title: str = "Memory Graph",
        save_dir: str = "graph_images",
        filename: str = ''
    ) -> str:
        """
        Display the graph using matplotlib with aesthetic enhancements and save it locally.
    
        Parameters:
            node_size (int): Size of the nodes.
            node_color_memories (str): Color for memory nodes.
            node_color_notes (str): Color for note nodes.
            tagged_node_colors (dict[str, str]): Tag names and their associated node colors.
            edge_color (str): Color for relationships.
            edge_color_has_note (str): Color for 'has_note' relationships.
            figsize (tuple[int, int]): Size of the matplotlib figure.
            with_labels (bool): Whether to display labels on nodes.
            with_edge_labels (bool): Whether to display labels on edges
            font_size (int): Font size for labels.
            title (str): Title of the graph.
            save_dir (str): Directory to save the graph image.
            filename (str): Specific filename for the saved image. If None, a unique filename is generated.
    
        Returns:
            Result (str): A string in the format "image:<path_to_image>".
        """
        plt.figure(figsize=figsize)
        G = self.graph.copy()
        if len(G.nodes) == 0:
            return "The graph hasn't been created yet"
        
        # Define node colors based on node type (memory, note, or tags)
        node_colors = []
        node_shapes = []
        for node_id, data in G.nodes(data=True):
            if 'title' in data:
                node_colors.append(node_color_memories)
                node_shapes.append('o')  # Circle for memories
            elif 'content' in data:
                node_colors.append(node_color_notes)
                node_shapes.append('s')  # Square for notes
            elif tagged_node_colors:
                matched_tag = None
                for tag, color in tagged_node_colors.items():
                    if tag in data.get('tags', []):  # Check if the node has the tag
                        matched_tag = color
                        break
                node_colors.append(matched_tag if matched_tag else 'grey')  # Default to grey if no tag matched
                node_shapes.append('^')  # Triangle for tagged nodes
            else:
                node_colors.append('grey')  # Default color
                node_shapes.append('^')  # Triangle for undefined types
    
        # Define edge colors based on relationship type
        edge_colors = []
        edge_relationship_types = set() 

        for u, v, data in G.edges(data=True):
            rel_type = data.get('relationship_type', 'relates_to')
            edge_colors.append(edge_color if rel_type != 'has_note' else edge_color_has_note)
            edge_relationship_types.add(rel_type)

        # Choose a layout for the graph
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
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
        if not with_edge_labels:
            for rel_type in edge_relationship_types:
                if rel_type == 'has_note':
                    legend_handles.append(mlines.Line2D([], [], color=edge_color_has_note, label='Has Note', linewidth=2))
                else:
                    legend_handles.append(mlines.Line2D([], [], color=edge_color, label=rel_type, linewidth=2))

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
                elif 'content' in data:
                    node_labels[node_id] = f"Note: {data['content'][:20]}..."  # Truncate long notes
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
        memory_patch = mpatches.Patch(color=node_color_memories, label='Memory')
        note_patch = mpatches.Patch(color=node_color_notes, label='Note')
        legend_handles = [memory_patch, note_patch]
        
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
        plt.axis('off')  # Hide the axes
        plt.tight_layout()
        
        image_dir = self.local_graph_dir / save_dir
        image_dir.mkdir(parents=True, exist_ok=True)

        # Generate a unique filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"memory_graph_{timestamp}.png"
        else:
            if not filename.endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
                filename += '.png'  # Default to PNG if no valid extension provided

        # Full path to save the image
        save_path = image_dir / filename

        # Save the figure
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        if os.path.exists(save_path):
            return f"image:{save_path}"
        # Return the image path as a URL string
        return "could not save graph"


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
                text_representation += f"{node_id}: {data}"
                print(f"{node_id}: {data}")
            print("\n--- Graph Edges ---")
            for u, v, data in self.graph.edges(data=True):
                text_representation += f"{u} --{data}--> {v}"
                print(f"{u} --{data}--> {v}")
        
        return text_representation

    @tool
    def build_structured_relationship(self, nodes: list[dict[str, Union[str, dict]]], relationships: list[dict]) -> dict[str, str]:
        """
        Saves an graph consisting of nodes and their relationships (edges).

        Args:
            nodes (list[dict]): A list of nodes.
                - title (str): Title of the node.
                - summary (str): Summary of the node.
                - tags (list[str]): Tags associated with the node.
                - notes (list[dict]): Notes related to the node.

            relationships (list[dict]): A list of relationships between node. 
                - from_node_title (str or None): Title of the source node.
                - to_node_title (str or None): Title of the target node.
                - relationship_type (str or None): Type of the relationship.

        Returns:
            node_id_map (dict[str, str]): A mapping of node titles to their unique IDs.

        """

        node_id_map = {}  # Maps memory titles to their unique IDs

        # Check for duplicates across the entire graph
        existing_titles = set()
        for node in nodes:
            if node['title'] in self.title_index and self.title_index[node['title']]:
                raise ValueError(f"Memory title '{node['title']}' already exists in the graph.")
            existing_titles.add(node['title'])

        with self.lock:
            # First, add all memories
            for node in nodes:
                for tag in node['tags']:
                    if tag not in self.tag_set:
                        self.add_tag(tag)
                # Add the memory
                memory_id = self.add_memory(
                    title=node['title'],
                    summary=node['summary'],
                    tags=node['tags'], 
                )
                node_id_map[node['title']] = memory_id

                # Add associated notes, if any
                for note in node['notes']:
                    self.add_note(
                        memory_id=memory_id,
                        content=note['content'],  
                        **note.get('metadata', {})
                    )

            # Then, establish relationships
            for relationship in relationships:
                from_title = relationship['from_memory_title']
                to_title = relationship['to_memory_title']
                rel_type = relationship['relationship_type']

                # Retrieve memory IDs from titles
                from_id = node_id_map.get(from_title)
                to_id = node_id_map.get(to_title)

                if not from_id:
                    raise ValueError(f"Memory with title '{from_title}' not found in provided memories.")
                if not to_id:
                    raise ValueError(f"Memory with title '{to_title}' not found in provided memories.")

                print(rel_type)
                # Add the relationship
                self.add_relationship(
                    from_memory_id=from_id,
                    to_memory_id=to_id,
                    relationship_type=rel_type,
                )

        return node_id_map
    
    ### Tag Management Methods ###
    def add_tag(self, tag: str) -> None:
        """
        Add a new tag to the tag set.

        Parameters:
            tag (str): The tag to be added.

        Raises:
            TagError: If the tag already exists or is invalid.
        """
        sanitized_tag = self._sanitize_tag(tag)
        with self.lock:
            self.tag_set.add(sanitized_tag)
            return f"Added tag: {sanitized_tag}"
        
        return f"Added tag: {tag}"

    @tool
    def list_tags(self) -> list[str]:
        """
        List all available tags.

        Returns:
            sorted_tags (list[str]): A list of all tags.
        """
        with self.lock:
            return sorted(list(self.tag_set))

    # ### Memory Management Methods ###
    @tool
    def add_memory(
        self,
        title: str,
        summary: str,
        tags: list[str],
        **metadata: any
    ) -> str:
        """
        Add a memory node with metadata and optional tags.

        Parameters:
            title (str): Title of the memory.
            summary (str): Summary or content of the memory.
            tags (list, optional): List of tags to associate with the memory.
            metadata (any): Additional metadata as key-value pairs.

        Returns:
            memory_id (str): The unique ID of the added memory.

        Raises:
            ValueError: If title or summary is invalid.
        """
        with self.lock:
            # Validate and sanitize inputs
            title = self._sanitize_input(title, max_length=255)
            summary = self._sanitize_input(summary, max_length=1024)
            sanitized_metadata = {k: self._sanitize_input(v) for k, v in metadata.items()}

            # Validate tags
            memory_tags = set()
            if tags:
                for tag in tags:
                    sanitized_tag = self._sanitize_tag(tag)
                    if sanitized_tag not in self.tag_set:
                        self.add_tag(sanitized_tag)
                    memory_tags.add(sanitized_tag)

            if title in self.title_index and self.title_index[title]:
                raise ValueError(f"A memory with the title '{title}' already exists.")

            # Generate a unique memory ID
            memory_id = str(uuid.uuid4())[:8]

            # Add the memory node
            self.graph.add_node(memory_id, title=title, summary=summary, tags=list(memory_tags), **sanitized_metadata)

            # Update indexes
            self.title_index[title].append(memory_id)
            for tag in memory_tags:
                self.tag_index[tag].add(memory_id)

            self.notifier.log(Rule(RULEPREFIX + f"Added Memory: {title}", style=RULESTYLE, align="left"))
            memory_details = f"**Title:** {title}\n**Summary:** {summary}\n**Tags:** {', '.join(tags)}\n**Metadata:** {metadata}"
            self.notifier.log(Markdown(memory_details))
            return memory_id

    @tool
    def update_memory(
        self,
        memory_id: str,
        title: str = '',
        summary: str = '',
        tags: list[str] = [],
        **metadata: any
    ) -> None:
        """
        Update an existing memory's details and tags.

        Parameters:
            memory_id (str): The ID of the memory to update.
            title (str, optional): New title.
            summary (str, optional): New summary.
            tags (list, optional): New list of tags.
            metadata (any): Additional metadata to update.

        Raises:
            MemoryNotFoundError: If the memory ID does not exist.
        """
        with self.lock:
            if memory_id not in self.graph:
                raise MemoryNotFoundError(f"Memory with ID '{memory_id}' not found.")

            node = self.graph.nodes[memory_id]

            # Update title
            if title:
                sanitized_title = self._sanitize_input(title, max_length=255)
                old_title = node['title']
                node['title'] = sanitized_title
                # Update title index
                self.title_index[old_title].remove(memory_id)
                if not self.title_index[old_title]:
                    del self.title_index[old_title]
                self.title_index[sanitized_title].append(memory_id)
                self.notifier.log(f"Memory ID={memory_id} title updated from '{old_title}' to '{sanitized_title}'")

            # Update summary
            if summary:
                node['summary'] = self._sanitize_input(summary, max_length=1024)
                self.notifier.log(f"Memory ID={memory_id} summary updated.")

            # Update metadata
            reserved_keys = {'title', 'summary', 'tags'}
            for key, value in metadata.items():
                if key in reserved_keys:
                    raise ValueError(f"Cannot overwrite reserved key '{key}' in metadata.")
                node[key] = self._sanitize_input(value)

            # Update tags
            if tags is not None:
                # Remove old tags from index
                old_tags = set(node.get('tags', []))
                for tag in old_tags:
                    self.tag_index[tag].discard(memory_id)
                    if not self.tag_index[tag]:
                        del self.tag_index[tag]

                # Validate and sanitize new tags
                new_tags = set()
                for tag in tags:
                    sanitized_tag = self._sanitize_tag(tag)
                    if sanitized_tag not in self.tag_set:
                        self.add_tag(sanitized_tag)
                    new_tags.add(sanitized_tag)

                # Update node tags
                node['tags'] = list(new_tags)

                # Update tag index
                for tag in new_tags:
                    self.tag_index[tag].add(memory_id)

                self.notifier.log(f"Memory ID={memory_id} tags updated to {list(new_tags)}")

    @tool
    def retrieve_memory(self, memory_id: str) -> dict[str, str]:
        """
        Retrieve a memory and its metadata by ID.

        Parameters:
            memory_id (str): The ID of the memory to retrieve.

        Returns:
           memory (dict[str, str]): The memory's data.

        Raises:
            MemoryNotFoundError: If the memory ID does not exist.
        """
        with self.lock:
            if memory_id in self.graph:
                return self.graph.nodes[memory_id]
            else:
                raise MemoryNotFoundError(f"Memory with ID '{memory_id}' not found.")

    @tool
    def get_memory_by_title(self, title: str) -> list:
        """
        Search and retrieve memories by title.

        Parameters:
            title (str): The title to search for.

        Returns:
            memories (list): A list of tuples containing memory IDs and their data.
        """
        with self.lock:
            sanitized_title = self._sanitize_input(title, max_length=255)
            node_ids = self.title_index.get(sanitized_title, [])
            return [(node_id, self.graph.nodes[node_id]) for node_id in node_ids] if node_ids else []

    @tool
    def search_memories_by_title_partial(self, search_term: str) -> list:
        """
        Search and retrieve memories by partial title match.

        Parameters:
            search_term (str): The substring to search for in titles.

        Returns:
            memories (list): A list of tuples containing memory IDs and their data.
        """
        with self.lock:
            pattern = re.compile(re.escape(search_term), re.IGNORECASE)
            matching_titles = filter(pattern.search, self.title_index.keys())
            result = []
            for title in matching_titles:
                for node_id in self.title_index[title]:
                    result.append((node_id, self.graph.nodes[node_id]))
            return result

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
        if not isinstance(depth, int) or depth < 1:
            raise ValueError("Depth must be a positive integer.")

        with self.lock:
            if node_id not in self.graph:
                raise MemoryNotFoundError(f"Node with ID '{node_id}' not found.")

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

    # ### Tag-Based Search Methods ###
    @tool
    def search_memories_by_tags(
        self,
        tags: list[str],
        match_all: bool = False
    ) -> list[tuple[str, dict[str, str]]]:
        """
        Search and retrieve memories by tags.

        Parameters:
            tags (list): List of tags to search for.
            match_all (bool): If True, only return memories that have all the specified tags.
                              If False, return memories that have any of the specified tags.

        Returns:
            memories (list[tuple[str, dict]]): A list of tuples containing memory IDs and their data.
        """
        with self.lock:
            sanitized_tags = [self._sanitize_tag(tag) for tag in tags]
            for tag in sanitized_tags:
                if tag not in self.tag_set:
                    return f"Tag '{tag}' does not exist. Use List tags"

            if not sanitized_tags:
                return []

            # Retrieve sets of memory IDs for each tag
            memory_sets = [self.tag_index[tag] for tag in sanitized_tags]

            # Compute intersection or union based on match_all
            if match_all:
                matched_memories = set.intersection(*memory_sets) if memory_sets else set()
            else:
                matched_memories = set.union(*memory_sets) if memory_sets else set()

            return [(memory_id, self.graph.nodes[memory_id]) for memory_id in matched_memories]

    # ### Note Management Methods ###
    @tool
    def add_note(
        self,
        memory_id: str,
        content: str,
        max_length: int = 2048,
        **metadata: any
    ) -> str:
        """
        Add a note as a separate node connected to a memory.

        Parameters:
            memory_id (str): The ID of the memory to attach the note to.
            content (str): The content of the note.
            max_length (int): Maximum allowed length of the note content.
            metadata (any): Additional metadata for the note.

        Returns:
            note_id (str): The unique ID of the added note.

        Raises:
            MemoryNotFoundError: If the memory ID does not exist.
            ValueError: If content is invalid.
        """
        with self.lock:
            if memory_id not in self.graph:
                raise MemoryNotFoundError(f"Memory with ID '{memory_id}' not found.")

            # Sanitize content and metadata
            content = self._sanitize_input(content, max_length=max_length)
            sanitized_metadata = {k: self._sanitize_input(v) for k, v in metadata.items()}

            # Generate a unique note ID
            note_id = str(uuid.uuid4())

            # Add the note node
            self.graph.add_node(note_id, content=content, tags=["note"], **sanitized_metadata)

            # Connect the note to the memory
            self.graph.add_edge(memory_id, note_id, relationship="has_note")

            self.notifier.log(f"Note added: ID={note_id} to Memory ID={memory_id}")

            return note_id

    @tool
    def get_notes_for_memory(self, memory_id: str) -> dict:
        """
        Retrieve all notes connected to a memory.

        Parameters:
            memory_id (str): The ID of the memory.

        Returns:
            notes (dict): A dictionary of note IDs and their data.

        Raises:
            MemoryNotFoundError: If the memory ID does not exist.
        """
        with self.lock:
            if memory_id in self.graph:
                notes = {}
                for neighbor in self.graph.neighbors(memory_id):
                    edge_data = self.graph.get_edge_data(memory_id, neighbor)
                    if edge_data.get('relationship') == 'has_note':
                        notes[neighbor] = self.graph.nodes[neighbor]
                return notes
            else:
                raise MemoryNotFoundError(f"Memory with ID '{memory_id}' not found.")

    # # Relationship Methods #
    @tool
    def add_lazy_relationship(self, from_node: dict, to_node: dict, relationship_type: str, **metadata):
        """
        Creates two new nodes and adds a relationship between them

        Parameters:
          from_node (dict): dictionary with keys title, summary, tags (list)
          to_node (dict): dictionary with keys tite, summary, tags (list)
          relationship_type (str): Specific relationship relating two nodes
          metadata (any): Additional {key: value} data to add to the relationship
        """
        from_node_id = self.add_memory(
            from_node["title"],
            from_node["summary"],
            from_node["tags"]
        )

        to_node_id = self.add_memory(
            to_node["title"],
            to_node["summary"],
            to_node["tags"]
        )

        self.graph.add_edge(
                from_node_id,
                to_node_id,
                relationship=relationship_type,
                **metadata
            )
        
        return f"created two nodes {from_node_id} and {to_node_id} with a {relationship_type.upper()} edge"

    @tool
    def add_relationship(
        self,
        from_memory_id: str,
        to_memory_id: str,
        relationship_type: str,
        **metadata: any
    ) -> None:
        """
        Add a relationship between two memories.

        Parameters:
            from_memory_id (str): The ID of the source memory.
            to_memory_id (str): The ID of the target memory.
            relationship_type (str): The type of relationship.
            metadata (any): Additional metadata for the relationship.

        Raises:
            MemoryNotFoundError: If either memory ID does not exist.
            ValueError: If attempting to create a relationship with invalid parameters.
        """
        with self.lock:
            if from_memory_id not in self.graph:
                raise MemoryNotFoundError(f"Memory with ID '{from_memory_id}' not found.")
            if to_memory_id not in self.graph:
                raise MemoryNotFoundError(f"Memory with ID '{to_memory_id}' not found.")
            if from_memory_id == to_memory_id:
                raise ValueError("Cannot create a relationship from a memory to itself.")
            
            # Add an edge with relationship data
            self.graph.add_edge(
                from_memory_id,
                to_memory_id,
                relationship=relationship_type,
            )
           
            self.notifier.log(f"Relationship '{relationship_type}' added from '{from_memory_id}' to '{to_memory_id}'.")
    
    @tool
    def remove_relationship(
        self,
        from_memory_id: str,
        to_memory_id: str,
        relationship_type: str = ''
    ) -> None:
        """
        Remove a specific relationship between two memories.

        Parameters:
            from_memory_id (str): The ID of the source memory.
            to_memory_id (str): The ID of the target memory.
            relationship_type (str, optional): The type of relationship to remove. If None, removes all relationships between the two memories.

        Raises:
            MemoryNotFoundError: If either memory ID does not exist.
            ValueError: If the specified relationship does not exist.
        """
        with self.lock:
            if from_memory_id not in self.graph:
                raise MemoryNotFoundError(f"Memory with ID '{from_memory_id}' not found.")
            if to_memory_id not in self.graph:
                raise MemoryNotFoundError(f"Memory with ID '{to_memory_id}' not found.")
            if not self.graph.has_edge(from_memory_id, to_memory_id):
                raise ValueError(f"No relationship exists between '{from_memory_id}' and '{to_memory_id}'.")
            
            # If a specific relationship type is provided, check and remove it
            if relationship_type:
                edge_data = self.graph.get_edge_data(from_memory_id, to_memory_id)
                if edge_data.get('relationship_type') != relationship_type:
                    raise ValueError(f"Relationship type '{relationship_type}' does not match the existing relationship.")
                self.graph.remove_edge(from_memory_id, to_memory_id)
            else:
                self.graph.remove_edge(from_memory_id, to_memory_id)
            
  
    @tool
    def get_relationships(self, memory_id: str) -> list:
        """
        Retrieve all relationships associated with a memory.

        Parameters:
            memory_id (str): The ID of the memory.

        Returns:
            relationships (list[dict]): A list of relationship details.

        Raises:
            MemoryNotFoundError: If the memory ID does not exist.
        """
        with self.lock:
            if memory_id not in self.graph:
                raise MemoryNotFoundError(f"Memory with ID '{memory_id}' not found.")
            relationships = []
            for neighbor in self.graph.neighbors(memory_id):
                edge_data = self.graph.get_edge_data(memory_id, neighbor)
                relationship = {
                    'to_memory_id': neighbor,
                    'relationship_type': edge_data.get('relationship_type', 'relates_to'),
                    'metadata': edge_data.get('metadata', {})
                }
                relationships.append(relationship)
            return relationships

    # ### Helper Methods ###
    def _sanitize_input(self, input_value: any, max_length: int = 255) -> str:
        """
        Sanitize input strings to prevent injection attacks and enforce length constraints.

        Parameters:
            input_value (any): The input to sanitize.
            max_length (int, optional): Maximum allowed length of the string.

        Returns:
            input (str): The sanitized string.

        Raises:
            ValueError: If the input is invalid or exceeds maximum length.
        """

        if not isinstance(input_value, str):
            input_value = str(input_value)
        if max_length is not None and len(input_value) > max_length:
            raise ValueError(
                f"Input exceeds maximum length of {max_length}."
                 "Try splitting the content into multiple memories, notes, or tags"
            )
        # Since HTML sanitization is not needed, perform basic sanitization
        sanitized = input_value.strip()
        if not sanitized:
            raise ValueError("Input cannot be empty or whitespace.")
        return sanitized
    
    def _ensure_graph_dirs(self) -> None:
        """Ensure memory directories exist"""
        self.local_graph_dir.parent.mkdir(parents=True, exist_ok=True)
        self.local_graph_dir.mkdir(exist_ok=True)
        self.global_graph_dir.parent.mkdir(parents=True, exist_ok=True)
        self.global_graph_dir.mkdir(exist_ok=True)

    def _sanitize_tag(self, tag: str) -> str:
        """
        Sanitize and standardize tag strings.

        Parameters:
            tag (str): The tag to sanitize.

        Returns:
            tag (str): The sanitized tag.

        Raises:
            TagError: If the tag is invalid.
        """
        sanitized_tag = self._sanitize_input(tag, max_length=50).lower()
        sanitized_tag = sanitized_tag.replace(' ', '_')
        return sanitized_tag

    def _current_timestamp(self) -> str:
        """
        Get the current UTC timestamp in ISO format.

        Returns:
            str: The current timestamp.
        """
        return datetime.now(timezone.utc).isoformat()


    @tool
    def save_graph(self, filename: str="memory_graph.pkl") -> str:
        """
        Save the graph as a pickle file.

        Parameters:
            filename (str): The file path to save the graph.

        Returns:
            filename (str): The filename saved

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
            return "Graph successfully saved."
        except IOError as e:
            # logger.error(f"Failed to save graph to {filename}: {e}")
            raise IOError(f"Failed to save graph: {e}")

    @tool
    def load_graph(self, filename) -> str:
        """
        Load the graph

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

                return "Graph loaded succesfully. You may now search the graph"
            
            # logger.info(f"Graph successfully loaded from {filename}.")
        except (IOError, pickle.UnpicklingError) as e:
            # logger.error(f"Failed to load graph from {filename}: {e}")
            raise IOError(f"Failed to load graph: {e}")
    
    def system(self) -> str:
        print("new")
        return f"""Use this toolkit to form long-term relationships.
        Learn from past experiences, and solve new problems faster.
        When encountering a new problem, search for related tags or memories to help create a plan. 
        """