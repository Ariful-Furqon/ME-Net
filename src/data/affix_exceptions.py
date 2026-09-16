import os

_LEXICON_PATH = os.path.join(os.path.dirname(__file__), "lexicon_root.txt")


def _load_roots():
    if not os.path.exists(_LEXICON_PATH):
        return set()
    with open(_LEXICON_PATH, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


_ROOTS = _load_roots()

# di-/ter-/te- initial words that are not passive or stative verbs.
PASSIVE_EXCEPTIONS = {r for r in _ROOTS if r.startswith(("di", "te"))} | {
    "dia", "diri", "dinas", "dingin", "dini", "dinding", "dimensi", "digital", "diskusi",
    "direktur", "direksi", "diplomat", "diplomasi", "distrik", "distribusi", "dinasti",
    "terus", "teman", "tepat", "tengah", "tempat", "tentara", "tentang", "tenaga",
    "teknologi", "teknis", "televisi", "telepon", "teror", "teroris", "terorisme",
    "tersangka", "terminal", "teritori", "tertib", "terbit", "teras", "tema", "tender",
}

# me-/ber-/be- initial words that are not active or intransitive verbs.
ACTIVE_NON_VERB_ROOTS = {r for r in _ROOTS if r.startswith(("me", "be"))} | {
    "mereka", "menteri", "memang", "menit", "media", "medis", "meja", "merah", "merek",
    "metode", "metro", "meter", "mesin", "mesjid", "menu", "berita", "besar", "benar",
    "bebas", "belum", "beras", "bensin", "bencana", "bendera", "berat", "betul", "bekas",
    "beberapa", "bersama", "belakang", "berapa", "bentuk", "beda", "bela", "beli",
}

# ke-...-an nominalisations whose base is not a bare root (derived or loan bases).
KE_AN_VALID = {
    "kebersamaan", "keberadaan", "keberhasilan", "kebersihan", "keberlanjutan",
    "kepresidenan", "kepolisian", "kementerian", "kejaksaan", "kedutaan", "kelurahan",
    "kecamatan", "kepercayaan", "keterlibatan", "ketidakpastian", "ketidakadilan",
    "kesejahteraan", "keselamatan", "kewarganegaraan", "kepemimpinan", "kepentingan",
    "kerusuhan", "kebakaran", "kecelakaan", "kerugian", "kekerasan", "kemerdekaan",
    "keputusan", "kesimpulan", "ketahanan", "ketenagakerjaan", "kepulauan", "keuangan",
}

# ke-...-an strings that are not nominalisations: lexicalised roots plus frequent false matches.
KE_AN_EXCEPTIONS = ({r for r in _ROOTS if r.startswith("ke") and r.endswith("an")} - KE_AN_VALID) | {
    "kemudian", "keluarkan", "kawan",
}

# Emphatic / interrogative bound clitics (possessive -nya is deliberately excluded).
BOUND_CLITICS = ["lah", "kah", "tah", "pun"]

# Free-standing (mostly colloquial) discourse particles.
FREE_PARTICLES = {
    "kok", "sih", "dong", "deh", "kan", "lho", "loh", "toh", "yah", "nah", "kek",
    "mah", "tuh", "nih", "dah", "lah", "pun", "kah", "wkwk", "wkwkwk",
}

# Words that end in a clitic string but are lexicalised (not base + clitic).
CLITIC_EXCEPTIONS = {
    "masalah", "sekolah", "jumlah", "salah", "allah", "wilayah", "kalah", "olah", "telah",
    "lelah", "sebelah", "istilah", "majalah", "nikah", "langkah", "sejarah", "ampun",
    "meskipun", "walaupun", "apapun", "siapapun", "manapun", "kapanpun", "bagaimanapun",
    "sedikitpun", "betapapun", "adapun", "ataupun", "sekalipun", "biarpun", "kendatipun",
    "lembah", "tanah", "arah", "sawah", "rumah", "sudah", "bawah", "tengah", "indah",
}
