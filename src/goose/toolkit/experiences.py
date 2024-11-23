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
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import matplotlib.pyplot as plt
import networkx as nx
from networkx import Graph
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

from goose.notifier import Notifier
from goose.toolkit.utils import RULEPREFIX, RULESTYLE

# Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)




class RelationshipType(str, Enum):
    # Causal Relationships
    CAUSES = "causes"
    LEADS_TO = "leads_to"
    RESULTS_IN = "results_in"
    ENABLES = "enables"
    FACILITATES = "facilitates"

    # Positive Relationships
    SUPPORTS = "supports"
    ENHANCES = "enhances"
    CONTRIBUTES_TO = "contributes_to"
    COMPLEMENTS = "complements"
    ENCOURAGES = "encourages"

    # Negative Relationships
    CONFLICTS_WITH = "conflicts_with"
    HINDERS = "hinders"
    DETRACTS_FROM = "detracts_from"
    OPPOSES = "opposes"
    UNDERMINES = "undermines"

    # Contextual Relationships
    OCCURS_BEFORE = "occurs_before"
    IS_PART_OF = "is_part_of"
    RELATES_TO = "relates_to"
    FOLLOWS = "follows"
    STEMS_FROM = "stems_from"
    SHARES_CONTEXT_WITH = "shares_context_with"

    # Hierarchical Relationships
    IS_SUBEVENT_OF = "is_subevent_of"
    BELONGS_TO = "belongs_to"
    FORMS_PART_OF = "forms_part_of"

    # Additional Relationships
    REPRESENTS = "represents"
    MIRRORS = "mirrors"
    INFORMS = "informs"
    GUIDES = "guides"
    EXEMPLIFIES = "exemplifies"


class Relationship(BaseModel):
    from_memory_title: Optional[str] = Field(None, max_length=255)
    to_memory_title: Optional[str] = Field(None, max_length=255)
    relationship_type: Optional[RelationshipType] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        extra = 'allow'
        arbitrary_types_allowed = True

    def to_dict(self):
        return {
            'from_memory_title': self.from_memory_title,
            'to_memory_title': self.to_memory_title,
            'relationship_type': self.relationship_type.name if self.relationship_type else None,
            'metadata': self.metadata
        }

class Note(BaseModel):
    content: str = Field(..., max_length=2048)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    note_id: str = Field(default_factory=lambda: str(uuid4()), exclude=True)

class Memory(BaseModel):
    title: str = Field(..., max_length=255)
    summary: str = Field(..., max_length=1024)
    tags: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    memory_id: str = Field(default_factory=lambda: str(uuid4()), exclude=True)
    notes: List[Note] = Field(default_factory=list)

class Experience(BaseModel):
    memories: List[Memory]
    relationships: Optional[List[Relationship]] = Field(default_factory=list)

class MemoryNotFoundError(Exception):
    pass


class TagError(Exception):
    def __init__(self, message: str, associated_memories: Optional[List[Dict[str, Any]]] = None):
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
        edge_color_related: str = 'gray',
        edge_color_leads_to: str = 'blue',
        edge_color_has_note: str = 'green',
        figsize: Tuple[int, int] = (12, 8),
        with_labels: bool = True,
        font_size: int = 10,
        title: Optional[str] = "Memory Graph",
        save_dir: str = "graph_images",
        filename: Optional[str] = None
    ) -> str:
        """
        Display the graph using matplotlib with aesthetic enhancements and save it locally.
    
        Parameters:
            node_size (int, optional): Size of the nodes.
            node_color_memories (str, optional): Color for memory nodes.
            node_color_notes (str, optional): Color for note nodes.
            edge_color_related (str, optional): Color for 'relates_to' relationships.
            edge_color_leads_to (str, optional): Color for 'leads_to' relationships.
            edge_color_has_note (str, optional): Color for 'has_note' relationships.
            figsize (tuple, optional): Size of the matplotlib figure.
            with_labels (bool, optional): Whether to display labels on nodes.
            font_size (int, optional): Font size for labels.
            title (str, optional): Title of the graph.
            save_dir (str, optional): Directory to save the graph image.
            filename (str, optional): Specific filename for the saved image. If None, a unique filename is generated.
    
        Returns:
            str: A string in the format "image: <path_to_image>".
        """
        plt.figure(figsize=figsize)
        G = self.graph.copy()
    
        # Define node colors based on node type (memory or note)
        node_colors = []
        node_shapes = []
        for node_id, data in G.nodes(data=True):
            if 'title' in data:
                node_colors.append(node_color_memories)
                node_shapes.append('o')  # Circle for memories
            elif 'content' in data:
                node_colors.append(node_color_notes)
                node_shapes.append('s')  # Square for notes
            else:
                node_colors.append('grey')  # Default color
                node_shapes.append('^')  # Triangle for undefined types
    
        # Define edge colors based on relationship type
        edge_colors = []
        for u, v, data in G.edges(data=True):
            rel_type = data.get('relationship_type', 'relates_to')
            if rel_type == 'relates_to':
                edge_colors.append(edge_color_related)
            elif rel_type == 'leads_to':
                edge_colors.append(edge_color_leads_to)
            elif rel_type == 'has_note':
                edge_colors.append(edge_color_has_note)
            else:
                edge_colors.append('black')  # Default edge color
    
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
                alpha=0.9,
                label='Memory' if shape == 'o' else 'Note' if shape == 's' else 'Other'
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
    
        # Draw labels
        if with_labels:
            labels = {}
            for node_id, data in G.nodes(data=True):
                if 'title' in data:
                    labels[node_id] = data['title']
                elif 'content' in data:
                    labels[node_id] = f"Note: {data['content'][:20]}..."  # Truncate long notes
                else:
                    labels[node_id] = node_id  # Fallback to node ID
            nx.draw_networkx_labels(
                G,
                pos,
                labels=labels,
                font_size=font_size,
                font_color='black'
            )
    
        # Create legend manually
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        # Define legend elements
        memory_patch = mpatches.Patch(color=node_color_memories, label='Memory')
        note_patch = mpatches.Patch(color=node_color_notes, label='Note')
        related_line = mlines.Line2D([], [], color=edge_color_related, label='Related To', linewidth=2)
        leads_to_line = mlines.Line2D([], [], color=edge_color_leads_to, label='Leads To', linewidth=2)
        has_note_line = mlines.Line2D([], [], color=edge_color_has_note, label='Has Note', linewidth=2)
    
        # Position legend outside the plot
        plt.legend(
            handles=[memory_patch, note_patch, related_line, leads_to_line, has_note_line],
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
    def print_graph(self):
        """
        Print the entire graph structure, including nodes and edges with their attributes.
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
    def add_experience(self, experience: Experience) -> Dict[str, str]:
        """
        Add multiple memories, relationships, and optional notes in a single operation.

        Parameters:
            experience (Experience): An Experience instance containing memories and relationships.

        Returns:
            dict: A mapping of memory titles to their unique IDs.

        Raises:
            TagError: If any of the provided tags do not exist.
            ValueError: If relationship definitions are invalid or duplicate titles are found.
        """
        memory_id_map = {}  # Maps memory titles to their unique IDs

        # Check for duplicates across the entire graph
        existing_titles = set()
        for memory in experience.memories:
            if memory.title in self.title_index and self.title_index[memory.title]:
                raise ValueError(f"Memory title '{memory.title}' already exists in the graph.")
            existing_titles.add(memory.title)

        with self.lock:
            # First, add all memories
            for memory in experience.memories:
                # Add the memory
                memory_id = self.add_memory(
                    title=memory.title,
                    summary=memory.summary,
                    tags=memory.tags,
                    **memory.metadata
                )
                memory_id_map[memory.title] = memory_id

                # Add associated notes, if any
                for note in memory.notes:
                    self.add_note(
                        memory_id=memory_id,
                        content=note.content,
                        **note.metadata
                    )

            # Then, establish relationships
            for relationship in experience.relationships:
                from_title = relationship.from_memory_title
                to_title = relationship.to_memory_title
                rel_type = relationship.relationship_type
                rel_metadata = relationship.metadata

                # Retrieve memory IDs from titles
                from_id = memory_id_map.get(from_title)
                to_id = memory_id_map.get(to_title)

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
                    **rel_metadata
                )

        return memory_id_map
    ### Tag Management Methods ###
    
    @tool
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
            if sanitized_tag in self.tag_set:
                raise TagError(f"Tag '{sanitized_tag}' already exists.")
            self.tag_set.add(sanitized_tag)
            self.tag_index[sanitized_tag]  # Initialize the tag index
            self.notifier.log(Rule(RULEPREFIX + f"Added Tag: {sanitized_tag}", style=RULESTYLE, align="left"))
            # self.audit_log.append({
            #     'action': 'add_tag',
            #     'tag': sanitized_tag,
            #     'timestamp': self._current_timestamp()
            # })
            # logger.info(f"Tag added: {sanitized_tag}")

    @tool
    def remove_tag(self, tag: str) -> None:
        """
        Remove a tag from the tag set.

        Parameters:
            tag (str): The tag to be removed.

        Raises:
            TagError: If the tag does not exist or is associated with memories.
        """
        sanitized_tag = self._sanitize_tag(tag)
        with self.lock:
            if sanitized_tag not in self.tag_set:
                raise TagError(f"Tag '{sanitized_tag}' does not exist.")
            
            associated_memories = self.search_memories_by_tags([sanitized_tag], match_all=True)
            print(associated_memories)
            if associated_memories:
                # Prepare a list of memory titles or IDs
                memory_details = [
                    {"memory_id": mem_id, "title": mem_data.get("title", "Untitled")}
                    for mem_id, mem_data in associated_memories
                ]
                raise TagError(
                    f"Tag '{sanitized_tag}' is associated with the following memories and cannot be removed.",
                    associated_memories=memory_details
                )
            
            # If no associated memories, proceed to remove the tag
            self.tag_set.remove(sanitized_tag)
            del self.tag_index[sanitized_tag]
            # self.audit_log.append({
            #     'action': 'remove_tag',
            #     'tag': sanitized_tag,
            #     'timestamp': self._current_timestamp()
            # })
            # logger.info(f"Tag removed: {sanitized_tag}")

    @tool
    def list_tags(self) -> List[str]:
        """
        List all available tags.

        Returns:
            list: A list of all tags.
        """
        with self.lock:
            return sorted(list(self.tag_set))

    ### Memory Management Methods ###
    @tool
    def add_memory(
        self,
        title: str,
        summary: str,
        tags: Optional[List[str]] = None,
        **metadata: Any
    ) -> str:
        """
        Add a memory node with metadata and optional tags.

        Parameters:
            title (str): Title of the memory.
            summary (str): Summary or content of the memory.
            tags (list, optional): List of tags to associate with the memory.
            metadata: Additional metadata as key-value pairs.

        Returns:
            str: The unique ID of the added memory.

        Raises:
            TagError: If any of the provided tags do not exist in the tag set.
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
                        raise TagError(f"Tag '{sanitized_tag}' does not exist. Please add it before using.")
                    memory_tags.add(sanitized_tag)

            if title in self.title_index and self.title_index[title]:
                raise ValueError(f"A memory with the title '{title}' already exists.")

            # Generate a unique memory ID
            memory_id = str(uuid.uuid4())

            # Add the memory node
            self.graph.add_node(memory_id, title=title, summary=summary, tags=list(memory_tags), **sanitized_metadata)

            # Update indexes
            self.title_index[title].append(memory_id)
            for tag in memory_tags:
                self.tag_index[tag].add(memory_id)

            self.notifier.log(Rule(RULEPREFIX + f"Added Memory: {title}", style=RULESTYLE, align="left"))
            memory_details = f"**Title:** {title}\n**Summary:** {summary}\n**Tags:** {', '.join(tags)}\n**Metadata:** {metadata}"
            self.notifier.log(Markdown(memory_details))
            # Audit log
            # self.audit_log.append({
            #     'action': 'add_memory',
            #     'memory_id': memory_id,
            #     'title': title,
            #     'tags': list(memory_tags),
            #     'timestamp': self._current_timestamp()
            # })
            # logger.info(f"Memory added: ID={memory_id}, Title='{title}'")

            return memory_id

    @tool
    def update_memory(
        self,
        memory_id: str,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **metadata: Any
    ) -> None:
        """
        Update an existing memory's details and tags.

        Parameters:
            memory_id (str): The ID of the memory to update.
            title (str, optional): New title.
            summary (str, optional): New summary.
            tags (list, optional): New list of tags.
            metadata: Additional metadata to update.

        Raises:
            MemoryNotFoundError: If the memory ID does not exist.
            TagError: If any of the provided tags do not exist in the tag set.
            ValueError: If attempting to overwrite reserved keys.
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
                # logger.info(f"Memory ID={memory_id} title updated from '{old_title}' to '{sanitized_title}'")

            # Update summary
            if summary:
                node['summary'] = self._sanitize_input(summary, max_length=1024)
                # logger.info(f"Memory ID={memory_id} summary updated.")

            # Update metadata
            reserved_keys = {'title', 'summary', 'tags'}
            for key, value in metadata.items():
                if key in reserved_keys:
                    raise ValueError(f"Cannot overwrite reserved key '{key}' in metadata.")
                node[key] = self._sanitize_input(value)
                # logger.info(f"Memory ID={memory_id} metadata '{key}' updated.")

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
                        raise TagError(f"Tag '{sanitized_tag}' does not exist. Please add it before using.")
                    new_tags.add(sanitized_tag)

                # Update node tags
                node['tags'] = list(new_tags)

                # Update tag index
                for tag in new_tags:
                    self.tag_index[tag].add(memory_id)

                # logger.info(f"Memory ID={memory_id} tags updated to {list(new_tags)}")

            # Audit log
            # self.audit_log.append({
            #     'action': 'update_memory',
            #     'memory_id': memory_id,
            #     'timestamp': self._current_timestamp()
            # })

    @tool
    def retrieve_memory(self, memory_id: str) -> Dict[str, Any]:
        """
        Retrieve a memory and its metadata by ID.

        Parameters:
            memory_id (str): The ID of the memory to retrieve.

        Returns:
            dict: The memory's data.

        Raises:
            MemoryNotFoundError: If the memory ID does not exist.
        """
        with self.lock:
            if memory_id in self.graph:
                return self.graph.nodes[memory_id]
            else:
                raise MemoryNotFoundError(f"Memory with ID '{memory_id}' not found.")

    @tool
    def get_memory_by_title(self, title: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Search and retrieve memories by title.

        Parameters:
            title (str): The title to search for.

        Returns:
            list: A list of tuples containing memory IDs and their data.
        """
        with self.lock:
            sanitized_title = self._sanitize_input(title, max_length=255)
            node_ids = self.title_index.get(sanitized_title, [])
            return [(node_id, self.graph.nodes[node_id]) for node_id in node_ids] if node_ids else []

    @tool
    def search_memories_by_title_partial(self, search_term: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Search and retrieve memories by partial title match.

        Parameters:
            search_term (str): The substring to search for in titles.

        Returns:
            list: A list of tuples containing memory IDs and their data.
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
    def view_memories_at_node(
        self,
        node_id: str,
        depth: int = 3,
        title_only: bool = False,
        relationship_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        View memories connected to a specific node up to a certain depth,
        optionally filtering by relationship types.

        Parameters:
            node_id (str): The ID of the starting node.
            depth (int, optional): The depth of traversal. Defaults to 3.
            title_only (bool, optional): If True, return only titles. Defaults to False.
            relationship_filter (list, optional): List of relationship types to include.

        Returns:
            dict: A dictionary of connected memories up to the specified depth.
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
                        edge_data = self.graph.get_edge_data(current_node, neighbor)
                        if relationship_filter:
                            if edge_data.get('relationship_type') not in relationship_filter:
                                continue
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, current_depth + 1))

            return result

    ### Tag-Based Search Methods ###
    @tool
    def search_memories_by_tags(
        self,
        tags: List[str],
        match_all: bool = False
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Search and retrieve memories by tags.

        Parameters:
            tags (list): List of tags to search for.
            match_all (bool): If True, only return memories that have all the specified tags.
                              If False, return memories that have any of the specified tags.

        Returns:
            list: A list of tuples containing memory IDs and their data.

        Raises:
            TagError: If any of the provided tags do not exist in the tag set.
        """
        with self.lock:
            sanitized_tags = [self._sanitize_tag(tag) for tag in tags]
            for tag in sanitized_tags:
                if tag not in self.tag_set:
                    raise TagError(f"Tag '{tag}' does not exist.")

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

    ### Note Management Methods ###
    @tool
    def add_note(
        self,
        memory_id: str,
        content: str,
        max_length: Optional[int] = 2048,
        **metadata: Any
    ) -> str:
        """
        Add a note as a separate node connected to a memory.

        Parameters:
            memory_id (str): The ID of the memory to attach the note to.
            content (str): The content of the note.
            max_length (int, optional): Maximum allowed length of the note content.
            metadata: Additional metadata for the note.

        Returns:
            str: The unique ID of the added note.

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
            self.graph.add_node(note_id, content=content, **sanitized_metadata)

            # Connect the note to the memory
            self.graph.add_edge(memory_id, note_id, relationship="has_note")

            # Audit log
            # self.audit_log.append({
            #     'action': 'add_note',
            #     'note_id': note_id,
            #     'memory_id': memory_id,
            #     'timestamp': self._current_timestamp()
            # })
            self.notifier.log(f"Note added: ID={note_id} to Memory ID={memory_id}")

            return note_id

    @tool
    def get_notes_for_memory(self, memory_id: str) -> Dict[str, Any]:
        """
        Retrieve all notes connected to a memory.

        Parameters:
            memory_id (str): The ID of the memory.

        Returns:
            dict: A dictionary of note IDs and their data.

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

    # Relationship Methods #
    @tool
    def add_relationship(
        self,
        from_memory_id: str,
        to_memory_id: str,
        relationship_type: RelationshipType,
        **metadata: Any
    ) -> None:
        """
        Add a relationship between two memories.

        Parameters:
            from_memory_id (str): The ID of the source memory.
            to_memory_id (str): The ID of the target memory.
            relationship_type (RelationshipType): The type of relationship (e.g., 'relates_to').
            metadata: Additional metadata for the relationship.

        Raises:
            MemoryNotFoundError: If either memory ID does not exist.
            ValueError: If attempting to create a relationship with invalid parameters.
        """
        # logger.debug(f"Adding relationship '{relationship_type}' from '{from_memory_id}' to '{to_memory_id}'")
        with self.lock:
            if from_memory_id not in self.graph:
                # logger.error(f"Source memory ID '{from_memory_id}' not found.")
                raise MemoryNotFoundError(f"Memory with ID '{from_memory_id}' not found.")
            if to_memory_id not in self.graph:
                # logger.error(f"Target memory ID '{to_memory_id}' not found.")
                raise MemoryNotFoundError(f"Memory with ID '{to_memory_id}' not found.")
            if from_memory_id == to_memory_id:
                # logger.error("Cannot create a relationship from a memory to itself.")
                raise ValueError("Cannot create a relationship from a memory to itself.")
            # Create the relationship instance
            relationship = Relationship(relationship_type=relationship_type, metadata=metadata)
            # Add an edge with relationship data
            self.graph.add_edge(from_memory_id, to_memory_id, **relationship.to_dict())
            # Audit log
            # self.audit_log.append({
            #     'action': 'add_relationship',
            #     'from_memory_id': from_memory_id,
            #     'to_memory_id': to_memory_id,
            #     'relationship_type': relationship_type.value,
            #     'metadata': metadata,
            #     'timestamp': self._current_timestamp()
            # })
            self.notifier.log(f"Relationship '{relationship_type}' added from '{from_memory_id}' to '{to_memory_id}'.")
    
    @tool
    def remove_relationship(
        self,
        from_memory_id: str,
        to_memory_id: str,
        relationship_type: Optional[str] = None
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
        # logger.debug(f"Removing relationship from '{from_memory_id}' to '{to_memory_id}' of type '{relationship_type}'")
        with self.lock:
            if from_memory_id not in self.graph:
                # logger.error(f"Source memory ID '{from_memory_id}' not found.")
                raise MemoryNotFoundError(f"Memory with ID '{from_memory_id}' not found.")
            if to_memory_id not in self.graph:
                # logger.error(f"Target memory ID '{to_memory_id}' not found.")
                raise MemoryNotFoundError(f"Memory with ID '{to_memory_id}' not found.")
            if not self.graph.has_edge(from_memory_id, to_memory_id):
                # logger.error(f"No relationship exists between '{from_memory_id}' and '{to_memory_id}'.")
                raise ValueError(f"No relationship exists between '{from_memory_id}' and '{to_memory_id}'.")
            
            # If a specific relationship type is provided, check and remove it
            if relationship_type:
                edge_data = self.graph.get_edge_data(from_memory_id, to_memory_id)
                if edge_data.get('relationship_type') != relationship_type:
                    # logger.error(f"Relationship type '{relationship_type}' does not match the existing relationship.")
                    raise ValueError(f"Relationship type '{relationship_type}' does not match the existing relationship.")
                self.graph.remove_edge(from_memory_id, to_memory_id)
                # logger.info(f"Relationship '{relationship_type}' removed from '{from_memory_id}' to '{to_memory_id}'.")
            else:
                # Remove all relationships between the two memories
                self.graph.remove_edge(from_memory_id, to_memory_id)
                # logger.info(f"All relationships removed from '{from_memory_id}' to '{to_memory_id}'.")
            
            # Audit log
            self.audit_log.append({
                'action': 'remove_relationship',
                'from_memory_id': from_memory_id,
                'to_memory_id': to_memory_id,
                'relationship_type': relationship_type,
                'timestamp': self._current_timestamp()
            })

    @tool
    def get_relationships(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all relationships associated with a memory.

        Parameters:
            memory_id (str): The ID of the memory.

        Returns:
            list: A list of relationship details.

        Raises:
            MemoryNotFoundError: If the memory ID does not exist.
        """
        # logger.debug(f"Retrieving relationships for memory ID '{memory_id}'")
        with self.lock:
            if memory_id not in self.graph:
                # logger.error(f"Memory ID '{memory_id}' not found.")
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
            # logger.debug(f"Found {len(relationships)} relationships for memory ID '{memory_id}'")
            return relationships

    ### Helper Methods ###
    @tool
    def _sanitize_input(self, input_value: Any, max_length: Optional[int] = 255) -> str:
        """
        Sanitize input strings to prevent injection attacks and enforce length constraints.

        Parameters:
            input_value (Any): The input to sanitize.
            max_length (int, optional): Maximum allowed length of the string.

        Returns:
            str: The sanitized string.

        Raises:
            ValueError: If the input is invalid or exceeds maximum length.
        """
        if isinstance(input_value, RelationshipType):
            input_value = input_value.value

        elif not isinstance(input_value, str):
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
            str: The sanitized tag.

        Raises:
            TagError: If the tag is invalid.
        """
        sanitized_tag = self._sanitize_input(tag, max_length=50).lower()
        sanitized_tag = sanitized_tag.replace(' ', '_')
        if not sanitized_tag or sanitized_tag == '_':
            raise TagError("Tag cannot be empty or consist only of spaces.")
        return sanitized_tag

    def _current_timestamp(self) -> str:
        """
        Get the current UTC timestamp in ISO format.

        Returns:
            str: The current timestamp.
        """
        return datetime.now(timezone.utc).isoformat()

    def audit_logs(self) -> List[Dict[str, Any]]:
        """
        Retrieve the audit logs.

        Returns:
            list: A list of audit log entries.
        """
        with self.lock:
            return list(self.audit_log)

    @tool
    def save_graph(self, filename: str="memory_graph.pkl") -> None:
        """
        Save the graph as a pickle file.

        Parameters:
            filename (str): The file path to save the graph.

        Raises:
            IOError: If the file cannot be written.
        """
        try:
            filename = self.global_graph_dir / filename
            with open(filename, 'wb') as file:
                pickle.dump({
                    'graph': self.graph,
                    'title_index': self.title_index,
                    'tag_set': self.tag_set,
                    'tag_index': self.tag_index,
                    'audit_log': self.audit_log,
                }, file)

            self.audit_log.append({
                'action': 'save_graph',
                'filename': filename,
                'timestamp': self._current_timestamp()
            })
            self.notifier.log(f"Graph successfully saved to {filename}.")
        except IOError as e:
            # logger.error(f"Failed to save graph to {filename}: {e}")
            raise IOError(f"Failed to save graph: {e}")

    @tool
    def load_graph(self) -> None:
        """
        Load the graph

        Raises:
            IOError: If the file cannot be read.
        """
        try:
            filename = self.global_graph_dir / "memory_graph.pkl"
            with open(filename, 'rb') as file:
                data = pickle.load(file)
                self.graph = data['graph']
                self.title_index = data['title_index']
                self.tag_set = data['tag_set']
                self.tag_index = data['tag_index']
                self.audit_log = data['audit_log']
            self.audit_log.append({
                'action': 'load_graph',
                'filename': filename,
                'timestamp': self._current_timestamp()
            })
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

# IDEA: should we save individual runs as a memory graph,
#  and then once completed we save the memory graph and connect it to the long term memory?