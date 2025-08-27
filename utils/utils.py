import boto3


def _chunk_even(seq, k: int):
    """Split seq into k chunks, as evenly sized as possible."""
    n = len(seq)
    k = max(1, min(k, n))
    q, r = divmod(n, k)
    out, i = [], 0
    for j in range(k):
        size = q + (1 if j < r else 0)
        out.append(seq[i : i + size])
        i += size
    return out

