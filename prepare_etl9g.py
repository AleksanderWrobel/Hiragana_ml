"""
Wyciąga hiraganę z ETL9G i zapisuje ją jako jeden plik .npz gotowy do treningu.

Użycie:
    python prepare_etl9g.py --src ./ETL9G --out hiragana_etl9g.npz --size 64

--src to katalog z rozpakowanymi plikami ETL9G_01 ... ETL9G_50.
"""
import argparse
import re
import struct
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

RECORD_SIZE = 8199          # bajtów na jeden rekord w ETL9G
IMG_W, IMG_H = 128, 127     # rozmiar obrazu, 4 bity na piksel (16 poziomów szarości)
# Format rekordu (big-endian):
# 2H: nr arkusza, kod JIS | 8s: czytanie | I: nr próbki | 4B, 4H, 2B: metadane
# 34x: pominięte | 8128s: obraz (128*127/2 bajtów) | 7x: pominięte
RECORD_FMT = ">2H8sI4B4H2B34x8128s7x"
assert struct.calcsize(RECORD_FMT) == RECORD_SIZE


def jis_to_char(jis: int) -> str:
    """JIS X 0208 -> znak Unicode (przez kodowanie EUC-JP)."""
    return bytes([(jis >> 8) | 0x80, (jis & 0xFF) | 0x80]).decode("euc_jp")


def is_hiragana(ch: str) -> bool:
    return "\u3041" <= ch <= "\u309f"


def decode_image(raw: bytes) -> np.ndarray:
    """Rozpakowuje 4-bitowe piksele do tablicy uint8 0-255 o kształcie (127, 128)."""
    b = np.frombuffer(raw, dtype=np.uint8)
    px = np.empty(b.size * 2, dtype=np.uint8)
    px[0::2] = b >> 4       # starszy półbajt = lewy piksel
    px[1::2] = b & 0x0F
    return (px.reshape(IMG_H, IMG_W) * 17).astype(np.uint8)   # 0..15 -> 0..255


def preprocess(img: np.ndarray, size: int) -> np.ndarray:
    """Dopełnia do kwadratu 128x128 i skaluje do size x size."""
    sq = np.zeros((IMG_W, IMG_W), dtype=np.uint8)
    sq[: IMG_H, :] = img
    return np.asarray(Image.fromarray(sq).resize((size, size), Image.LANCZOS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("hiragana_etl9g.npz"))
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()

    files = sorted(p for p in args.src.iterdir() if re.fullmatch(r"ETL9G_\d+", p.name))
    if not files:
        raise SystemExit(f"Brak plików ETL9G_xx w {args.src}")

    images, chars, file_idx, sheets = [], [], [], []
    for p in files:
        data = p.read_bytes()
        if len(data) % RECORD_SIZE:
            raise SystemExit(f"{p.name}: rozmiar {len(data)} nie dzieli się przez {RECORD_SIZE}")
        fid = int(p.name.split("_")[1])
        n_kept = 0
        for off in range(0, len(data), RECORD_SIZE):
            r = struct.unpack_from(RECORD_FMT, data, off)
            sheet, jis, raw = r[0], r[1], r[-1]
            try:
                ch = jis_to_char(jis)
            except UnicodeDecodeError:
                continue
            if not is_hiragana(ch):
                continue
            images.append(preprocess(decode_image(raw), args.size))
            chars.append(ch)
            file_idx.append(fid)
            sheets.append(sheet)
            n_kept += 1
        print(f"{p.name}: {len(data) // RECORD_SIZE} rekordów, hiragana: {n_kept}")

    classes = sorted(set(chars))
    to_id = {c: i for i, c in enumerate(classes)}
    labels = np.array([to_id[c] for c in chars], dtype=np.int64)

    np.savez_compressed(
        args.out,
        images=np.stack(images),                 # (N, size, size) uint8
        labels=labels,                           # (N,) int64
        classes=np.array(classes),               # nazwa klasy (znak) dla każdego id
        file_idx=np.array(file_idx, np.int16),   # z którego pliku: do podziału po piszących
        sheet=np.array(sheets, np.int32),        # nr arkusza (pisarza)
    )

    counts = Counter(chars)
    print(f"\nZapisano {len(labels)} obrazów, {len(classes)} klas -> {args.out}")
    print("Klasy:", "".join(classes))
    print(f"Próbek na klasę: min {min(counts.values())}, max {max(counts.values())}")


if __name__ == "__main__":
    main()
