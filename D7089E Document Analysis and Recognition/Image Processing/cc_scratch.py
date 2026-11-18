import numpy as np
from collections import deque

N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N8 = N4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def label_components(binary, connectivity=4):
    """Flood-fill labelling: returns (num_labels incl. background, label image)."""
    offsets = N4 if connectivity == 4 else N8
    H, W = binary.shape
    labels = np.zeros((H, W), np.int32)
    current = 0
    for r in range(H):
        for c in range(W):
            if binary[r, c] and labels[r, c] == 0:   # an unlabelled "on" pixel = new component
                current += 1
                labels[r, c] = current
                queue = deque([(r, c)])
                while queue:                          # breadth-first search over neighbours
                    y, x = queue.popleft()
                    for dy, dx in offsets:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and binary[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            queue.append((ny, nx))
    return current + 1, labels
