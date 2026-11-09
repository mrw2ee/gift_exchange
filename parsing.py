
import re

def parse_pairings(text):
    """
    Parse assignment text into a list of (giver, receiver) tuples.

    Expected format:
        Assignments:
            Alice is giving to Bob
            Bob is giving to Charlie

    Returns:
        List of (giver, receiver) tuples.
    """

    pairings = []

    pattern = re.compile(
        r"^\s*(.*?)\s+is giving to\s+(.*?)\s*$"
    )

    for line in text.splitlines():
        match = pattern.match(line)

        if match:
            giver = match.group(1).strip()
            receiver = match.group(2).strip()

            pairings.append((giver, receiver))

    return pairings
