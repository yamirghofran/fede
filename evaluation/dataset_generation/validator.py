"""
Lexical Leakage Detection and Validation
Checks generated queries for unwanted mentions of movie titles, actor names, etc.
"""

import re
import logging
from typing import Dict, List, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryValidator:
    """
    Validates generated queries for lexical leakage.
    Detects mentions of movie titles, actor names, and other forbidden content.
    """

    # Common words that should be ignored (short stopwords, pronouns, prepositions, etc.)
    # These are excluded from validation to avoid false positives
    COMMON_WORDS_BLACKLIST = {
        "a",
        "an",
        "the",
        "it",
        "is",
        "in",
        "on",
        "at",
        "by",
        "to",
        "for",
        "of",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "and",
        "or",
        "but",
        "as",
        "if",
        "so",
        "no",
        "go",
        "do",
        "be",
        "me",
        "my",
        "we",
        "us",
        "he",
        "him",
        "she",
        "her",
        "they",
        "them",
        "one",
        "two",
        "six",
        "ten",
        "big",
        "bad",
        "new",
        "old",
        "red",
        "yes",
        "now",
        "day",
        "way",
        "all",
        "any",
        "are",
        "can",
        "did",
        "has",
        "had",
        "how",
        "let",
        "may",
        "not",
        "our",
        "own",
        "say",
        "see",
        "too",
        "use",
        "war",
        "who",
        "why",
        "you",
        "bit",
        "cut",
        "end",
        "eye",
        "far",
        "few",
        "got",
        "hit",
        "hot",
        "job",
        "joy",
        "key",
        "law",
        "lie",
        "low",
        "man",
        "map",
        "mix",
        "net",
        "pay",
        "put",
        "ran",
        "run",
        "sad",
        "set",
        "sin",
        "sit",
        "sun",
        "tax",
        "tea",
        "tie",
        "top",
        "try",
        "van",
        "war",
        "win",
        "yet",
    }

    # Minimum length for movie titles to be considered in validation
    MIN_TITLE_LENGTH = 3

    def __init__(self, metadata: Dict, strictness: str = "medium"):
        """
        Initialize validator with movie metadata.

        Args:
            metadata: Dictionary of all movie metadata
            strictness: Validation strictness (low, medium, high)
        """
        self.metadata = metadata
        self.strictness = strictness
        self.forbidden_words = self._build_forbidden_list()

    def _build_forbidden_list(self) -> Dict[str, Set[str]]:
        """
        Build lists of forbidden words from metadata.
        Filters out short titles and common words to reduce false positives.

        Returns:
            Dictionary of forbidden word categories
        """
        movie_titles = set()
        actor_names = set()

        for movie_data in self.metadata.values():
            # Add movie title (full title only to avoid false positives)
            file_info = movie_data.get("file", {})
            title = file_info.get("name", "")
            if title:
                title_lower = title.lower()
                # Skip short titles and common words to reduce false positives
                if (
                    len(title_lower) >= self.MIN_TITLE_LENGTH
                    and title_lower not in self.COMMON_WORDS_BLACKLIST
                ):
                    movie_titles.add(title_lower)

        # Note: We don't have explicit actor names in metadata
        # If available, we would add them here

        return {"movie_titles": movie_titles, "actor_names": actor_names}

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Simple tokenization - can be enhanced
        return re.findall(r"\b\w+\b", text.lower())

    def check_lexical_leakage(self, query: str, movie_title: str) -> Dict:
        """
        Check if query contains lexical leakage.

        Args:
            query: Generated query text
            movie_title: Target movie title

        Returns:
            Dictionary with leakage analysis
        """
        query_tokens = self._tokenize(query)
        title_tokens = self._tokenize(movie_title)

        # Check for direct title mention
        direct_match = movie_title.lower() in query.lower()

        # Check for partial title mentions
        partial_matches = []
        for token in query_tokens:
            # Skip short words and common words (e.g., "the", "a", "an")
            if (
                token in title_tokens
                and len(token) > 2
                and token not in self.COMMON_WORDS_BLACKLIST
            ):
                partial_matches.append(token)

        # Check against forbidden titles (other movies) - use word boundaries for exact matching
        other_movie_matches = []
        query_lower = query.lower()
        for title in self.forbidden_words["movie_titles"]:
            if title == movie_title.lower():
                continue
            # Use word boundary matching to avoid substring false positives
            # \b matches at word boundaries (e.g., "it" won't match "spirit")
            pattern = r"\b" + re.escape(title) + r"\b"
            if re.search(pattern, query_lower, re.IGNORECASE):
                other_movie_matches.append(title)

        # Calculate leakage score
        leakage_score = 0.0
        if direct_match:
            leakage_score = 1.0
        elif partial_matches:
            leakage_score = min(0.8, len(partial_matches) * 0.3)
        elif other_movie_matches:
            leakage_score = min(0.6, len(other_movie_matches) * 0.2)

        # Determine if query should be flagged
        thresholds = {"low": 0.8, "medium": 0.5, "high": 0.3}
        is_flagged = leakage_score >= thresholds[self.strictness]

        return {
            "has_leakage": is_flagged,
            "leakage_score": leakage_score,
            "direct_match": direct_match,
            "partial_matches": partial_matches,
            "other_movie_matches": other_movie_matches,
            "reason": self._get_reason(
                direct_match, partial_matches, other_movie_matches
            ),
        }

    def _get_reason(
        self, direct_match: bool, partial_matches: List, other_movie_matches: List
    ) -> str:
        """Get human-readable reason for leakage flag."""
        if direct_match:
            return "Direct title match detected"
        elif partial_matches:
            return f"Partial title matches: {', '.join(partial_matches)}"
        elif other_movie_matches:
            return f"Other movie name matches: {', '.join(other_movie_matches)}"
        return "No leakage detected"

    @staticmethod
    def get_movie_name(query: Dict) -> str:
        """Return the movie name from a query dict, handling both dataset formats.

        Supports:
          - eval_queries.json format: ``movie_title`` field
          - generated_queries.json format: ``movie_name`` field
        """
        return query.get("movie_name") or query.get("movie_title", "")

    def validate_batch(self, queries: List[Dict]) -> Dict:
        """
        Validate a batch of queries and generate report.

        Args:
            queries: List of query dictionaries (either format - see get_movie_name)

        Returns:
            Validation report dictionary
        """
        results = {
            "total": len(queries),
            "flagged": 0,
            "passed": 0,
            "flagged_queries": [],
            "leakage_scores": [],
        }

        for query in queries:
            movie_name = self.get_movie_name(query)
            validation_result = self.check_lexical_leakage(
                query["query"], movie_name
            )

            # Add validation result to query
            query["validation"] = validation_result

            if validation_result["has_leakage"]:
                results["flagged"] += 1
                results["flagged_queries"].append(
                    {
                        "id": query["id"],
                        "query": query["query"],
                        "movie_name": query["movie_name"],
                        "leakage_score": validation_result["leakage_score"],
                        "reason": validation_result["reason"],
                    }
                )
            else:
                results["passed"] += 1

            results["leakage_scores"].append(validation_result["leakage_score"])

        # Calculate statistics
        if results["leakage_scores"]:
            results["avg_leakage_score"] = sum(results["leakage_scores"]) / len(
                results["leakage_scores"]
            )
        else:
            results["avg_leakage_score"] = 0.0

        return results
