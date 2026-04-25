# boilerplate.py
# ThemeDrift
#
# Centralized stopword list for 10-K TF-IDF filtering.
#
# Usage:
#   from boilerplate import get_boilerplate
#
#   words = get_boilerplate(source="manual")    # original hardcoded list
#   words = get_boilerplate(source="llm")       # GPT-generated list (requires OPENAI_API_KEY)
#   words = get_boilerplate(source="combined")  # union of both (recommended)
#
# The LLM result is cached to data/boilerplate_llm_cache.json so the API is
# only called once. Pass force=True to regenerate.

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

CACHE_PATH = Path("data/boilerplate_llm_cache.json")

# Original hardcoded list
MANUAL = [

    # === 1. legal / filing boilerplate ===
    "item", "items", "statements", "form", "sec",
    "financial", "company", "business", "including",
    "related", "information", "management",
    "annual", "report", "fiscal", "year", "quarter", "period",
    "thereof", "herein", "hereof", "hereby",
    "whereas", "notwithstanding", "accordance",

    # === 2. proxy / governance ===
    "board", "committee",
    "shareholder", "shareholders",
    "stockholder", "stockholders",
    "meeting", "vote", "voting",
    "amendment",

    # === 3. executive / titles ===
    "officer", "director", "president", "vice",
    "chief", "executive", "senior",

    # === 4. proxy / filing keywords ===
    "proxy", "pursuant", "duly", "signed",
    "incorporated", "reference",

    # === 5. signature / registrant block ===
    "registrant", "behalf", "principal", "accounting",
    "executed", "undersigned",
    "signature", "signatures",
    "corporate", "persons", "caused",
    "chairman", "authorized", "indicated",
    "capacity", "capacities",

    # === 6. forward-looking / PSLRA ===
    "forward", "looking",
    "historical",
    "expression", "expressions", "similar",
    "projection", "projections",
    "act", "securities", "reform", "private",
    "meaning", "constitute",

    # === 7. forward-looking verbs ===
    "believe", "believes",
    "expect", "expects",
    "anticipate", "anticipates",
    "plan", "plans",
    "intend", "intends",
    "seek", "seeks",
    "estimate", "estimates",

    # === 8. disclosure / narrative ===
    "relating", "related",
    "involve", "involves",
    "outlook",
    "assumption", "assumptions",
    "disclosure", "disclosures",
    "matter", "matters",

    # === 9. risk / litigation ===
    "litigation", "factor", "factors",
    "development", "developments",
    "expectation", "expectations", "regarding",
    "uncertainty", "uncertainties",
    "risk", "risks",

    # === 10. trademark / naming ===
    "trademark", "trademarks", "name", "names",

    # === 11. months / time ===
    "january", "february", "march", "april",

    # === 12. misc formatting ===
    "page", "pages", "applicable", "xa", "jr", "mr", "ms", "mrs",

    # === 13. personal names (sample, not exhaustive) ===
    "catherine", "jamie", "miller", "james", "john", "robert", "michael",
    "william", "david", "thomas", "richard", "charles", "joseph", "christopher",
    "daniel", "matthew", "anthony", "mark", "donald", "steven", "paul", "andrew",
    "kenneth", "george", "brian", "edward", "kevin", "ronald", "timothy",
    "mary", "patricia", "linda", "barbara", "elizabeth", "jennifer", "maria",
    "susan", "margaret", "dorothy", "lisa", "nancy", "karen", "betty",
    "lloyd", "lawrence", "dean",

    # === 14. filing / signature ===
    "exchange", "corporation",
    "director", "directors",
    "requirement", "requirements",
    "statement",

    # === 15. forward-looking ===
    "future", "futures",
    "result", "results",
    "fact", "facts",
    "caution",
    "subject",
]


# LLM-generated list
# Strategy: use a two-message conversation.
#   - system: strict role + output contract
#   - user:   the word list request
# This reliably keeps gpt-4o-mini from generating explanations.
_SYSTEM_PROMPT = (
    "You are a JSON API. You output only raw JSON. "
    "No markdown, no explanation, no preamble, no postamble. "
    "Your entire response must be a single valid JSON object."
)

_USER_PROMPT = """\
Generate a stopword list for filtering SEC 10-K annual report filings.
Include words from:
- Legal boilerplate: pursuant, herein, thereof, whereas, notwithstanding
- SEC structure: item, exhibit, registrant, form, filing
- Forward-looking: anticipate, outlook, projection, estimate, believe
- Executive titles: chief, officer, chairman, president, vice
- Governance: stockholder, amendment, quorum, proxy, voting
- Signature block: undersigned, executed, duly, signed, behalf
- Risk factor: uncertainty, factor, material, litigation, disclosure
- Common US first names (james, john, robert, michael, william, david, etc.)
- Common US last names (smith, johnson, williams, brown, jones, miller, etc.)
- Time references: fiscal, quarterly, annual, period, quarter
- Generic business: business, operations, management, financial, company

Output this JSON object and nothing else:
{"words": ["word1", "word2", ...]}

Include 200-220 unique lowercase words total.
"""


def _get_api_key():
    # 1. try config.py first
    try:
        from config import OPENAI_API_KEY
        if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-..."):
            return OPENAI_API_KEY
    except ImportError:
        pass
    # 2. fallback to environment variable
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError(
            "No API key found. Set OPENAI_API_KEY in config.py or environment."
        )
    return key


def generate_llm(model: str = "gpt-4o-mini", force: bool = False) -> list[str]:
    """
    Call the OpenAI API to generate a boilerplate stopword list.

    Requires OPENAI_API_KEY in config.py or environment.
    Result is cached to CACHE_PATH; pass force=True to regenerate.
    """
    # cache hit
    if CACHE_PATH.exists() and not force:
        log.info(f"Loading LLM boilerplate from cache: {CACHE_PATH}")
        with open(CACHE_PATH) as f:
            data = json.load(f)
        return data["words"]

    # call API
    api_key = _get_api_key()
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    log.info(f"Calling OpenAI ({model}) to generate boilerplate stopwords ...")
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_PROMPT},
        ],
        temperature=0.0,   # deterministic — less chance of runaway generation
        max_tokens=8192,   # 200 words as JSON strings ~ 600 tokens; 2048 is 3x headroom
    )

    finish_reason = response.choices[0].finish_reason
    raw = response.choices[0].message.content.strip()
    log.info(f"  finish_reason: {finish_reason}  |  response length: {len(raw)} chars")

    # guard: if truncated, raise a clear error rather than silent bad parse
    if finish_reason == "length":
        raise RuntimeError(
            f"Model response was truncated (finish_reason=length). "
            f"Response was {len(raw)} chars. "
            f"This should not happen with max_tokens=2048 and 200 words. "
            f"First 200 chars of raw:\n{raw[:200]}"
        )

    # strip markdown fences just in case
    if "```" in raw:
        raw = "\n".join(
            line for line in raw.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    # parse
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        words = parsed.get("words", [])
    elif isinstance(parsed, list):
        words = parsed
    else:
        raise ValueError(
            f"Unexpected JSON structure: {type(parsed)}\nRaw:\n{raw[:300]}"
        )

    if not words:
        raise ValueError(f"LLM returned empty word list.\nRaw:\n{raw[:500]}")

    # normalize: lowercase, strip whitespace, deduplicate
    words = sorted(
        set(w.lower().strip() for w in words if isinstance(w, str) and w.strip())
    )
    log.info(f"  LLM returned {len(words)} stopwords")

    # cache result
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump({"model": model, "words": words}, f, indent=2)
    log.info(f"  Cached to {CACHE_PATH}")

    return words


# Public API
def get_boilerplate(
    source: str = "manual",
    llm_model: str = "gpt-4o-mini",
    force_llm: bool = False,
) -> list[str]:
    """
    Return the boilerplate stopword list.

    Args:
        source:    "manual"   - original hardcoded list (default, no API needed)
                   "llm"      - GPT-generated list (requires OPENAI_API_KEY)
                   "combined" - union of manual + LLM (recommended)
        llm_model: OpenAI model (default: gpt-4o-mini)
        force_llm: bypass cache and regenerate if True
    """
    if source == "manual":
        return list(MANUAL)

    if source == "llm":
        return generate_llm(model=llm_model, force=force_llm)

    if source == "combined":
        manual_set = set(MANUAL)
        llm_words  = generate_llm(model=llm_model, force=force_llm)
        llm_new    = [w for w in llm_words if w not in manual_set]
        log.info(
            f"  Combined: {len(MANUAL)} manual + {len(llm_new)} LLM-only = "
            f"{len(MANUAL) + len(llm_new)} total"
        )
        return list(MANUAL) + llm_new

    raise ValueError(
        f"Unknown source '{source}'. Choose from: 'manual', 'llm', 'combined'"
    )


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    mode = sys.argv[1] if len(sys.argv) > 1 else "manual"
    words = get_boilerplate(source=mode)
    print(f"\nSource: {mode}")
    print(f"Total stopwords: {len(words)}")
    print(f"Sample (first 20): {words[:20]}")