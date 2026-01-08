import re
from enum import IntEnum, auto

class Subgoal(IntEnum):
    NO_OP = 0
    GOTO_OBJECT = 1 # Generic "Go to <obj>"
    PICK_OBJECT = 2
    OPEN_DOOR = 3
    DROP_OBJECT = 4
    EXPLORE = 5
    # Add more as needed

# Canonical mapping for structured representation
# (Action, Color, Object) -> Subgoal ID
# Note: This is a simplified mapping for Phase 0.
# We will rely on text parsing to extract these attributes.

def parse_subgoal(text):
    """
    Parses LLM output text into a canonical (Tuple, ID) format.

    Args:
        text (str): The raw text output from the LLM.

    Returns:
        tuple: ( (action, color, object), subgoal_id )
    """
    text = text.lower().strip()

    # Defaults
    action_type = "no_op"
    color = "any"
    obj_type = "none"
    subgoal_id = Subgoal.NO_OP

    # Regex Patterns
    # "pick up the key" or "pick yellow key"
    if re.search(r"pick.*key", text):
        action_type = "pick"
        obj_type = "key"
        subgoal_id = Subgoal.PICK_OBJECT
        # extraction of color is brittle with simple regex, assume 'any' or specific if mentioned
        if "yellow" in text: color = "yellow"
        elif "red" in text: color = "red"
        elif "green" in text: color = "green"
        elif "blue" in text: color = "blue"

    elif re.search(r"pick.*ball", text):
        action_type = "pick"
        obj_type = "ball"
        subgoal_id = Subgoal.PICK_OBJECT
        if "yellow" in text: color = "yellow"
        elif "red" in text: color = "red"
        elif "green" in text: color = "green"
        elif "blue" in text: color = "blue"
        elif "purple" in text: color = "purple"

    elif re.search(r"open.*door", text):
        action_type = "open"
        obj_type = "door"
        subgoal_id = Subgoal.OPEN_DOOR
        if "yellow" in text: color = "yellow"
        elif "red" in text: color = "red"
        elif "green" in text: color = "green"
        elif "blue" in text: color = "blue"

    elif re.search(r"go.*to.*door", text) or re.search(r"reach.*door", text):
        action_type = "goto"
        obj_type = "door"
        subgoal_id = Subgoal.GOTO_OBJECT
        if "yellow" in text: color = "yellow"
        elif "red" in text: color = "red"
        elif "green" in text: color = "green"
        elif "blue" in text: color = "blue"

    elif re.search(r"go.*to.*key", text) or re.search(r"reach.*key", text):
        action_type = "goto"
        obj_type = "key"
        subgoal_id = Subgoal.GOTO_OBJECT
        if "yellow" in text: color = "yellow"
        elif "red" in text: color = "red"
        elif "green" in text: color = "green"
        elif "blue" in text: color = "blue"

    elif re.search(r"go.*to.*goal", text) or re.search(r"reach.*goal", text):
        action_type = "goto"
        obj_type = "goal"
        subgoal_id = Subgoal.GOTO_OBJECT
        color = "green" # goal is usually green in minigrid

    # Fallback to NO_OP if nothing matched

    return (action_type, color, obj_type), subgoal_id.value
