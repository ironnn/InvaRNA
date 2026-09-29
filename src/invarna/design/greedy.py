"""Greedy constrained search over a caller-provided mutation neighborhood."""


def greedy_search(initial, neighbors, score, is_valid=lambda _: True, max_steps=100):
    current = initial
    current_score = float(score(current))
    for _ in range(max_steps):
        candidates = [candidate for candidate in neighbors(current) if is_valid(candidate)]
        if not candidates:
            break
        best = max(candidates, key=score)
        best_score = float(score(best))
        if best_score <= current_score:
            break
        current, current_score = best, best_score
    return current, current_score
