# Vlastní dataset – focení 14. 11. 2026

**Sestava:** iPhone, zoom 1,5×, telefon 30 cm nad podložkou, kolmo dolů. Tmavá matná podložka,
pravítko 30 cm u horního okraje záběru (kalibrace px → mm).

**Rozlišení:** všechny fotky jsou zmenšené na 1428×1071 px, takže mají stejný počet pixelů na mm.
Originály (5712×4284 a 4032×3024 px, celkem 235 MB) nejsou v gitu, leží na sdíleném Drive
a lokálně v `data/raw/`.

| Složka | Počet | Obsah |
|---|---|---|
| `samostatne/` | 35 | vždy jen jedna hodnota mince, `<hodnota>kc_<lic/rub>_<n>.jpg`; 1 Kč 3 ks, 2 a 5 Kč 5 ks, 10/20/50 Kč 4 ks na snímek (50 Kč: 2× líc, 3× rub) |
| `mix/` | 21 | různé hodnoty volně od sebe; sousední snímky jsou často skoro stejná scéna |
| `dotyk/` | 10 | dotýkající se mince |
| `test/` | 16 | testovací sada (nepoužívat k ladění), z toho 2 prázdné podložky |
| `robust/` | 6 | boční světlo (baterka) |

Líc = strana se lvem a nápisem ČESKÁ REPUBLIKA, rub = strana s hodnotou a motivem.

**Soubory:**
- `anotace.csv` – pro každou fotku počet mincí podle hodnot, celkový počet a součet v Kč.
- `anotace_mince.csv` – poloha (x, y, poloměr v px) a hodnota každé mince; ručně ověřeno.
- `puvodni_nazvy.csv` – převod na původní názvy z iPhonu (`IMG_xxxx.JPG`).

**Poznámky:**
- `test/test_02.jpg` je rozmazaná a má jiný výřez; jedna mince v ní je nejistá (20 vs 10 Kč).
- Sousední snímky v `mix/` a `dotyk/` se často liší jen posunutím jedné mince, proto trénovací
  a validační sadu dělíme po skupinách scén, ne po jednotlivých fotkách.
