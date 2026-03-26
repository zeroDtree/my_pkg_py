def longest_common_substring(a, b):
    """Return (length, start_in_a, start_in_b) of the longest common substring.

    Uses a rolling two-row DP table so memory is O(min(n, m)) instead of O(n*m).
    """
    # Keep the shorter string as the column dimension to minimise memory.
    if len(a) < len(b):
        a, b = b, a
        swapped = True
    else:
        swapped = False

    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)

    best_len = 0
    best_pos_a = 0
    best_pos_b = 0

    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                curr[j + 1] = prev[j] + 1
                if curr[j + 1] > best_len:
                    best_len = curr[j + 1]
                    best_pos_a = i - best_len + 1
                    best_pos_b = j - best_len + 1
            else:
                curr[j + 1] = 0
        prev, curr = curr, prev
        # Reuse the old `prev` buffer as the new `curr` without allocation.
        for k in range(m + 1):
            curr[k] = 0

    if swapped:
        best_pos_a, best_pos_b = best_pos_b, best_pos_a

    return best_len, best_pos_a, best_pos_b


def diff_regions_by_lcs(s1, s2):
    """Return a list of (s1_start, s1_end, s2_start, s2_end) diff regions.

    Uses an explicit stack instead of recursion to avoid hitting Python's
    recursion limit on long strings, and avoids string slicing by tracking
    sub-ranges via index offsets.
    """
    results = []
    # Each stack frame: (s1_slice, s2_slice, offset1, offset2)
    stack = [(s1, s2, 0, 0)]

    while stack:
        t1, t2, off1, off2 = stack.pop()

        lcs_len, pos1, pos2 = longest_common_substring(t1, t2)

        if lcs_len == 0:
            if t1 or t2:
                s1_end = off1 + len(t1) - 1 if t1 else off1 - 1
                s2_end = off2 + len(t2) - 1 if t2 else off2 - 1
                results.append((off1, s1_end, off2, s2_end))
            continue

        # Right part first so left part is processed first (LIFO order).
        stack.append(
            (
                t1[pos1 + lcs_len :],
                t2[pos2 + lcs_len :],
                off1 + pos1 + lcs_len,
                off2 + pos2 + lcs_len,
            )
        )
        stack.append(
            (
                t1[:pos1],
                t2[:pos2],
                off1,
                off2,
            )
        )

    # Results come out in left-to-right order because of the LIFO push order.
    return results


def _merge_ranges(ranges):
    """Merge overlapping or adjacent (start, end) ranges.

    Assumes `ranges` is already sorted by start position.
    """
    merged = []
    for s, e in ranges:
        if not merged or s > merged[-1][1] + 1:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return merged


def find_all_differences_with_context(s1, s2, context_size=2):
    diff_ranges = diff_regions_by_lcs(s1, s2)

    if not diff_ranges:
        return "", ""

    s1_ext = []
    s2_ext = []
    len_s1 = len(s1)
    len_s2 = len(s2)

    for s1_start, s1_end, s2_start, s2_end in diff_ranges:
        s1_ext.append((max(0, s1_start - context_size), min(len_s1 - 1, s1_end + context_size)))
        s2_ext.append((max(0, s2_start - context_size), min(len_s2 - 1, s2_end + context_size)))

    # diff_ranges is already sorted, so s1_ext / s2_ext are too — skip sorted().
    merged_s1 = _merge_ranges(s1_ext)
    merged_s2 = _merge_ranges(s2_ext)

    s1_parts = [s1[s : e + 1] for s, e in merged_s1]
    s2_parts = [s2[s : e + 1] for s, e in merged_s2]

    return "".join(s1_parts), "".join(s2_parts)


if __name__ == "__main__":
    s1 = "eeeeeHell123o, eeeeewwwwwworld!"
    s2 = "eeeeeHel2222lo, eeeeewwwwwworld!1"
    print(find_all_differences_with_context(s1, s2, context_size=2))
