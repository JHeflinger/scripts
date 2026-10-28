"""
split_stl.py - Split an STL file into its separate (disconnected) bodies.

Triangles are grouped into bodies by connectivity: two triangles belong to the
same body if they share a vertex (after welding vertices that are within a
small tolerance of each other). Each body is written out as its own binary STL.

Usage:
    python split_stl.py input.stl
    python split_stl.py input.stl -o out_dir --tol 1e-5 --min-tris 10

Requires: numpy (scipy is used if available for speed, but is optional).
"""

import argparse
import os
import struct
import sys

import numpy as np


# --------------------------------------------------------------------------- #
# STL reading
# --------------------------------------------------------------------------- #

def _is_binary_stl(path):
    size = os.path.getsize(path)
    if size < 84:
        return False
    with open(path, "rb") as f:
        f.seek(80)
        (count,) = struct.unpack("<I", f.read(4))
    # A binary STL has exactly 84 + 50 * count bytes. Checking this is more
    # reliable than looking for "solid", since many binary files start with it.
    return size == 84 + 50 * count


def _read_binary_stl(path):
    dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("verts", "<f4", (3, 3)),
        ("attr", "<u2"),
    ])
    with open(path, "rb") as f:
        header = f.read(80)
        (count,) = struct.unpack("<I", f.read(4))
        data = np.fromfile(f, dtype=dtype, count=count)
    return header, data["verts"].astype(np.float64), data["normal"].astype(np.float64)


def _read_ascii_stl(path):
    verts = []
    normals = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "facet" and len(parts) >= 5:
                normals.append([float(x) for x in parts[2:5]])
            elif parts[0] == "vertex" and len(parts) >= 4:
                verts.append([float(x) for x in parts[1:4]])
    verts = np.asarray(verts, dtype=np.float64)
    if len(verts) % 3 != 0:
        raise ValueError("ASCII STL vertex count is not a multiple of 3")
    tris = verts.reshape(-1, 3, 3)
    normals = np.asarray(normals, dtype=np.float64)
    if len(normals) != len(tris):
        normals = np.zeros((len(tris), 3))
    return b"", tris, normals


def read_stl(path):
    """Return (header, triangles[N,3,3], normals[N,3])."""
    if _is_binary_stl(path):
        return _read_binary_stl(path)
    return _read_ascii_stl(path)


# --------------------------------------------------------------------------- #
# STL writing
# --------------------------------------------------------------------------- #

def write_binary_stl(path, tris, normals, name="body"):
    tris = np.asarray(tris, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)

    # Recompute any missing/zero normals from the winding order.
    bad = np.linalg.norm(normals, axis=1) < 1e-12
    if np.any(bad):
        e1 = tris[bad, 1] - tris[bad, 0]
        e2 = tris[bad, 2] - tris[bad, 0]
        n = np.cross(e1, e2)
        lens = np.linalg.norm(n, axis=1, keepdims=True)
        lens[lens == 0] = 1.0
        normals[bad] = n / lens

    dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("verts", "<f4", (3, 3)),
        ("attr", "<u2"),
    ])
    data = np.zeros(len(tris), dtype=dtype)
    data["normal"] = normals
    data["verts"] = tris

    header = name.encode("ascii", "replace")[:80].ljust(80, b" ")
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(tris)))
        data.tofile(f)


# --------------------------------------------------------------------------- #
# Connectivity analysis
# --------------------------------------------------------------------------- #

def weld_vertices(tris, tol):
    """
    Map every triangle corner to a shared vertex index. Corners whose
    coordinates round to the same grid cell (size = tol) are merged.
    Returns faces[N,3] of vertex indices and the vertex count.
    """
    corners = tris.reshape(-1, 3)
    if tol > 0:
        keys = np.round(corners / tol).astype(np.int64)
    else:
        keys = corners
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    return inverse.reshape(-1, 3), int(inverse.max()) + 1 if len(inverse) else 0


def _components_scipy(faces, n_verts):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    # Edges between the corners of each triangle (a-b, b-c is enough to link all 3).
    rows = np.concatenate([faces[:, 0], faces[:, 1]])
    cols = np.concatenate([faces[:, 1], faces[:, 2]])
    data = np.ones(len(rows), dtype=np.int8)
    graph = coo_matrix((data, (rows, cols)), shape=(n_verts, n_verts))
    _, vert_labels = connected_components(graph, directed=False)
    return vert_labels


def _components_union_find(faces, n_verts):
    parent = np.arange(n_verts)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    for a, b, c in faces.tolist():
        ra, rb, rc = find(a), find(b), find(c)
        if ra != rb:
            parent[rb] = ra
        if ra != rc:
            parent[find(rc)] = ra

    roots = np.array([find(i) for i in range(n_verts)])
    _, vert_labels = np.unique(roots, return_inverse=True)
    return vert_labels


def find_bodies(tris, tol=1e-6):
    """
    Return a list of index arrays, one per body, each holding the indices of
    the triangles in that body. Sorted largest body first.
    """
    faces, n_verts = weld_vertices(tris, tol)
    if n_verts == 0:
        return []
    try:
        vert_labels = _components_scipy(faces, n_verts)
    except ImportError:
        vert_labels = _components_union_find(faces, n_verts)

    tri_labels = vert_labels[faces[:, 0]]
    order = np.argsort(tri_labels, kind="stable")
    sorted_labels = tri_labels[order]
    splits = np.flatnonzero(np.diff(sorted_labels)) + 1
    bodies = np.split(order, splits)
    bodies.sort(key=len, reverse=True)
    return bodies


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def describe(tris):
    pts = tris.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    # Signed volume via divergence theorem (meaningful for closed, consistently wound meshes)
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    volume = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0
    return lo, hi, volume


def main(argv=None):
    p = argparse.ArgumentParser(description="Split an STL into one STL per separate body.")
    p.add_argument("input", help="input .stl file (ASCII or binary)")
    p.add_argument("-o", "--out-dir", default=None,
                   help="output directory (default: <input name>_bodies next to the input)")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="vertex weld tolerance in model units (default 1e-6); "
                        "raise it if bodies that should be one come out split")
    p.add_argument("--min-tris", type=int, default=1,
                   help="skip bodies with fewer triangles than this (filters debris)")
    p.add_argument("--prefix", default=None, help="output file name prefix (default: input name)")
    args = p.parse_args(argv)

    header, tris, normals = read_stl(args.input)
    if len(tris) == 0:
        print("No triangles found in input.", file=sys.stderr)
        return 1

    base = os.path.splitext(os.path.basename(args.input))[0]
    prefix = args.prefix or base
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.input)), base + "_bodies")
    os.makedirs(out_dir, exist_ok=True)

    bodies = find_bodies(tris, args.tol)
    kept = [b for b in bodies if len(b) >= args.min_tris]

    print(f"Read {len(tris)} triangles from {args.input}")
    print(f"Found {len(bodies)} separate bodies"
          + (f" ({len(bodies) - len(kept)} skipped below --min-tris)" if len(kept) != len(bodies) else ""))

    width = max(3, len(str(len(kept))))
    for i, idx in enumerate(kept, 1):
        name = f"{prefix}_body_{i:0{width}d}"
        path = os.path.join(out_dir, name + ".stl")
        write_binary_stl(path, tris[idx], normals[idx], name=name)
        lo, hi, vol = describe(tris[idx])
        size = hi - lo
        print(f"  [{i:>{width}}] {len(idx):>8} tris  "
              f"size {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f}  "
              f"vol {abs(vol):.3f}  -> {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
