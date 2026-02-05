def longest_common_substring(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    best_len = 0
    best_pos_a = 0
    best_pos_b = 0

    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
                if dp[i + 1][j + 1] > best_len:
                    best_len = dp[i + 1][j + 1]
                    best_pos_a = i - best_len + 1
                    best_pos_b = j - best_len + 1

    return best_len, best_pos_a, best_pos_b


def diff_regions_by_lcs(s1, s2, offset1=0, offset2=0):
    lcs_len, pos1, pos2 = longest_common_substring(s1, s2)
    if lcs_len == 0:
        if s1 or s2:
            s1_end = offset1 + len(s1) - 1 if s1 else offset1 - 1
            s2_end = offset2 + len(s2) - 1 if s2 else offset2 - 1
            return [(offset1, s1_end, offset2, s2_end)]
        else:
            return []

    results = []

    results += diff_regions_by_lcs(s1[:pos1], s2[:pos2], offset1, offset2)

    results += diff_regions_by_lcs(
        s1[pos1 + lcs_len :], s2[pos2 + lcs_len :], offset1 + pos1 + lcs_len, offset2 + pos2 + lcs_len
    )

    return results


def find_all_differences_with_context(s1, s2, context_size=2):
    diff_ranges = diff_regions_by_lcs(s1, s2)

    if not diff_ranges:
        return [], []

    s1_ext = []
    s2_ext = []

    for s1_start, s1_end, s2_start, s2_end in diff_ranges:
        ext_s1_start = max(0, s1_start - context_size)
        ext_s1_end = min(len(s1) - 1, s1_end + context_size)
        s1_ext.append((ext_s1_start, ext_s1_end))

        ext_s2_start = max(0, s2_start - context_size)
        ext_s2_end = min(len(s2) - 1, s2_end + context_size)
        s2_ext.append((ext_s2_start, ext_s2_end))

    def merge_ranges(ranges):
        merged = []
        for s, e in sorted(ranges):
            if not merged or s > merged[-1][1] + 1:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        return merged

    merged_s1 = merge_ranges(s1_ext)
    merged_s2 = merge_ranges(s2_ext)

    s1_parts = [s1[s : e + 1] for s, e in merged_s1]
    s2_parts = [s2[s : e + 1] for s, e in merged_s2]

    return "".join(s1_parts), "".join(s2_parts)


if __name__ == "__main__":
    s1 = "eeeeeHell123o, eeeeewwwwwworld!"
    s2 = "eeeeeHel2222lo, eeeeewwwwwworld!1"
    print(find_all_differences_with_context(s1, s2, context_size=2))
