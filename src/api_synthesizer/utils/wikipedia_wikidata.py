#
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
#

import logging
from collections import deque

import requests
import wikipediaapi
from SPARQLWrapper import JSON, SPARQLWrapper


logger = logging.getLogger('api_synthesizer')


def get_wikidata_qid(domain_name: str) -> str:
    """Retrieve the Wikidata QID for a given domain name.

    Queries the Wikidata SPARQL endpoint to find the entity that matches
    the provided domain name label in English.

    Args:
        domain_name: The English label to search for in Wikidata.

    Returns:
        The Wikidata QID (e.g., 'Q58199') if found, None otherwise.
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(f"""
        SELECT ?item WHERE {{
            ?item rdfs:label "{domain_name}"@en .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 1
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    bindings = results["results"]["bindings"]
    if bindings:
        qid = bindings[0]["item"]["value"].split("/")[-1]
        logger.debug(f"[{domain_name}] Resolved Wikidata QID: {qid}")
        return qid

    logger.warning(f"[{domain_name}] Could not resolve Wikidata QID")
    return None


def fetch_entity(qid: str) -> dict:
    """Fetch full Wikidata entity JSON for a given QID.

    Retrieves the complete entity data from Wikidata's EntityData API.

    Args:
        qid: The Wikidata QID (e.g., 'Q58199' for Instant messaging).

    Returns:
        A dictionary containing the full Wikidata entity JSON response.

    Raises:
        requests.HTTPError: If the HTTP request fails.
    """
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    headers = {"User-Agent": "tool-dialogue-synthesizer/1.0 (your_email@example.com)"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()


class WikidataEntityExplorer:
    """Explorer class for retrieving and analyzing Wikidata entity information.

    This class provides methods to fetch and explore various properties and relationships
    of a Wikidata entity, including labels, descriptions, classifications, and hierarchies.

    Attributes:
        entity_id: The Wikidata QID of the entity being explored.
        entity: The raw entity data dictionary from Wikidata, or None if loading failed.
        label: The English label of the entity, or None if not available.
        description: The English description of the entity, or None if not available.
    """
    def __init__(self, entity_id: str) -> None:
        """Initialize the explorer with a Wikidata entity ID.

        Fetches the entity data from Wikidata and extracts the label and description.
        If fetching fails, the entity attributes are set to None.

        Args:
            entity_id: The Wikidata QID (e.g., 'Q58199') to explore.
        """
        logger.debug(f"Initializing explorer for {entity_id}...")
        self.entity_id = entity_id
        try:
            data = fetch_entity(self.entity_id)
            self.entity = data["entities"][self.entity_id]
            self.label = self.entity["labels"]["en"]["value"]
            self.description = self.entity.get("descriptions", {}).get("en", {}).get("value")
            logger.debug(f"Successfully loaded entity: {self.label} ({self.entity_id})")
        except Exception:
            logger.exception(f"Could not load entity {entity_id}.")
            self.entity = None
            self.label = None
            self.description = None


    def get_label_and_description(self) -> tuple[str, str] | tuple[None, None]:
        """Get the label and description of the entity.

        Returns:
            A tuple of (label, description) if the entity exists, otherwise (None, None).
        """
        if not self.entity:
            return None, None
        return self.label, self.description


    def get_classes(self) -> list[str]:
        """Get the classification labels for the entity.

        Retrieves 'instance of' (P31) or 'subclass of' (P279) relationships
        and returns the English labels of the related entities.

        Returns:
            A list of class labels. Returns an empty list if the entity doesn't exist
            or has no classifications.
        """
        if not self.entity:
            return []
        class_labels = []
        claims = self.entity.get("claims", {})
        for prop in ("P31", "P279"):
            if prop in claims:
                for claim in claims[prop]:
                    try:
                        qid = claim["mainsnak"]["datavalue"]["value"]["id"]
                        parent_data = fetch_entity(qid)
                        label = parent_data["entities"][qid]["labels"]["en"]["value"]
                        class_labels.append(label)
                    except Exception:
                        continue
        return class_labels


    def get_parents_as_properties(self) -> list[tuple[str, str]]:
        """Get parent class relationships as property-value tuples.

        Retrieves 'subclass of' (P279) relationships and returns them as tuples
        of ("subclass of", parent_label).

        Returns:
            A list of tuples containing property name and parent label. Returns an
            empty list if the entity doesn't exist or has no parent relationships.
        """
        if not self.entity:
            return []
        properties = []
        claims = self.entity.get("claims", {})
        if "P279" in claims:
            for claim in claims["P279"]:
                try:
                    value_qid = claim["mainsnak"]["datavalue"]["value"]["id"]
                    parent_data = fetch_entity(value_qid)
                    value_label = parent_data["entities"][value_qid]["labels"]["en"]["value"]
                    properties.append(("subclass of", value_label))
                except Exception:
                    continue
        return properties


    def get_all_properties(self, limit: int = 20) -> list[tuple[str, str]]:
        """Retrieve arbitrary property-value pairs from the entity.

        Fetches various properties (claims) associated with the entity and converts
        them to human-readable label-value pairs. Handles different value types
        including entity references, time values, and quantities.

        Args:
            limit: Maximum number of property groups to retrieve. Default is 20.

        Returns:
            A list of tuples containing (property_label, value_string). Returns an
            empty list if the entity doesn't exist.
        """
        if not self.entity:
            return []

        properties = []
        claims = self.entity.get("claims", {})
        count = 0
        for prop_id, claim_group in claims.items():
            if count >= limit:
                break

            try:
                prop_data = fetch_entity(prop_id)
                prop_label = prop_data["entities"][prop_id]["labels"]["en"]["value"]
            except Exception:
                continue

            for claim in claim_group:
                datavalue = claim.get("mainsnak", {}).get("datavalue")
                if not datavalue:
                    continue

                v = datavalue.get("value")
                vtype = datavalue.get("type")
                if vtype == "wikibase-entityid":
                    try:
                        vqid = v["id"]
                        vdata = fetch_entity(vqid)
                        vstr = vdata["entities"][vqid]["labels"]["en"]["value"]
                    except Exception:
                        vstr = v["id"]

                elif vtype == "time":
                    vstr = v.get("time", "Unknown Time")

                elif vtype == "quantity":
                    vstr = v.get("amount", "Unknown Amount")

                else:
                    vstr = str(v)
                properties.append((prop_label, vstr))
            count += 1

        return properties


    def get_upward_hierarchy(self, max_depth: int = 3) -> list[str]:
        """Traverse upward through the entity hierarchy using subclass and part-of relationships.

        Performs a breadth-first search upward through 'subclass of' (P279) and 
        'part of' (P361) relationships to build a hierarchical view of parent entities.

        Args:
            max_depth: Maximum depth to traverse in the hierarchy. Default is 3.

        Returns:
            A list of parent entity labels found in the hierarchy. Returns an empty
            list if the entity doesn't exist.
        """
        if not self.entity:
            return []
        hierarchy = []
        queue = deque([(self.entity_id, 0)])
        visited = {self.entity_id}

        while queue:
            current_qid, depth = queue.popleft()
            if depth >= max_depth:
                continue

            try:
                current_data = fetch_entity(current_qid)
                current_entity = current_data["entities"][current_qid]
            except Exception:
                continue

            claims = current_entity.get("claims", {})
            for prop in ("P279", "P361"):
                if prop in claims:
                    for claim in claims[prop]:
                        try:
                            parent_qid = claim["mainsnak"]["datavalue"]["value"]["id"]
                            if parent_qid not in visited:
                                parent_data = fetch_entity(parent_qid)
                                parent_label = parent_data["entities"][parent_qid]["labels"]["en"]["value"]
                                hierarchy.append(parent_label)
                                visited.add(parent_qid)
                                queue.append((parent_qid, depth + 1))
                        except Exception:
                            continue

        return hierarchy


def get_wikipedia_summary(topic: str) -> str:
    """Retrieve the summary section of a Wikipedia page for a given topic.

    Fetches the Wikipedia page for the specified topic and returns its summary.
    If the page does not exist, returns a default message.

    Args:
        topic: The title of the Wikipedia page to retrieve.

    Returns:
        The summary text of the Wikipedia page if it exists, otherwise 
        "No Wikipedia summary found."
    """
    wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='tool-dialogue-synthesizer/1.0 (email@example.com)'
        # Leaving this as a placeholder for now, interested users can fill in their info
    )
    page = wiki.page(topic)
    if page.exists():
        logger.debug(f"Retrieved summary for '{topic}'")
        return page.summary
    else:
        logger.warning(f"No page found for '{topic}'")
        return "No Wikipedia summary found."


def get_wikidata_subclasses(qid: str) -> list[str]:
    """Retrieve direct subclasses of a Wikidata entity.

    Queries the Wikidata SPARQL endpoint to find all entities that are direct
    subclasses (P279) of the specified entity.

    Args:
        qid: The Wikidata QID (e.g., 'Q58199') to find subclasses for.

    Returns:
        A list of English labels for entities that are subclasses of the given QID.
        Returns an empty list if the query fails or no subclasses are found.
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(f"""
        SELECT ?item ?itemLabel WHERE {{
            ?item wdt:P279 wd:{qid} .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
    """)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        return [r["itemLabel"]["value"] for r in results["results"]["bindings"] if "itemLabel" in r]
    except Exception:
        logger.exception("An error occurred while fetching subclasses.")
        return []


def get_wikipedia_title_from_wikidata(entity_json: dict, lang: str = "en") -> str | None:
    """Extract the Wikipedia page title from a Wikidata entity JSON.

    Searches the entity's sitelinks for the Wikipedia article in the specified language.

    Args:
        entity_json: The raw Wikidata entity JSON dictionary.
        lang: The language code for the Wikipedia site (default is "en").

    Returns:
        The Wikipedia page title if a sitelink exists for the specified language,
        otherwise None.
    """
    sitelinks = entity_json.get("sitelinks", {})
    wiki_key = f"{lang}wiki"
    if wiki_key in sitelinks:
        return sitelinks[wiki_key]["title"]
    return None


def build_domain_prompt(domain_name: str, qid: str) -> str:
    """Build a comprehensive prompt containing domain information from Wikipedia and Wikidata.

    Constructs a formatted prompt that includes Wikipedia summary, Wikidata entity
    information, classifications, subclasses, and properties for a given domain.

    Args:
        domain_name: The name of the domain to build the prompt for.
        qid: The Wikidata QID associated with the domain.

    Returns:
        A formatted string containing structured information about the domain,
        including Wikipedia summary and Wikidata metadata sections.
    """
    logger.info(f"Building prompt for domain: '{domain_name}' with QID: {qid}")

    #wiki_summary = get_wikipedia_summary(domain_name)
    explorer = WikidataEntityExplorer(qid)
    wiki_summary = get_wikipedia_summary(domain_name)

    if not explorer.entity:
        return "\n".join([
            "",
            "=== DOMAIN DESCRIPTION FROM WIKIPEDIA ===",
            wiki_summary,
            "",
        ])

    entity_label, entity_description = explorer.get_label_and_description()
    prompt_parts = [
        "\n=== DOMAIN DESCRIPTION FROM WIKIPEDIA ===",
        wiki_summary,
        f"\n=== DOMAIN INFORMATION FROM WIKIDATA ({qid}) ===",
        f"Label: {entity_label}",
        f"Description: {entity_description}",
    ]

    entity_classes = explorer.get_classes()
    if entity_classes:
        prompt_parts.append("\n=== CLASSIFICATION (The entity is an instance or subclass of) ===")
        prompt_parts.extend([f"- {c}" for c in entity_classes])

    subclasses = get_wikidata_subclasses(qid)
    if subclasses:
        prompt_parts.append("\n=== DIRECT SUBCLASSES (More specific types of the entity) ===")
        prompt_parts.extend([f"- {s}" for s in subclasses])

    entity_properties = explorer.get_parents_as_properties()
    if entity_properties:
        prompt_parts.append("\n=== SAMPLE PROPERTIES & RELATIONSHIPS (Attributes of the entity) ===")
        prompt_parts.extend([f"- {prop}: {value}" for prop, value in entity_properties])

    return "\n".join(prompt_parts)
