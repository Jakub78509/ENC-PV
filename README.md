# ENC-PV – Počítání mincí

Semestrální projekt do předmětu **Počítačové vidění (PV)**, Mendelova univerzita v Brně, ZS 2026/27.
Autoři: Jakub Procházka (xproch40), Jan Kostrhun.

## Cíl

Aplikace z kamery nad podložkou rozpozná české mince (1, 2, 5, 10, 20, 50 Kč),
u každé zobrazí hodnotu a v rohu obrazu jejich součet. Ostatní předměty a cizí mince zahodí.

## Hlavní vylepšení: stabilní živý součet

Součet v rohu obrazu se mění jen tehdy, když se opravdu změní mince na podložce, a mezi snímky neposkakuje.
Ukázka: položíme 10 mincí, součet drží stálou hodnotu; jednu minci odebereme a součet do zhruba půl sekundy
klesne o její hodnotu.

1. **Medián součtu v čase** – součet je medián z posledních N snímků (např. 15 ≈ 0,5 s); filtrace v časové doméně (př. 02).
2. **Sledování mincí** – každá mince má mezi snímky své ID (přiřazení podle nejbližšího středu), hodnota se určí hlasováním přes její historii.
3. **Hystereze** – nová mince se započítá až po k snímcích v řadě, chybějící se odebere až po m snímcích (stejný princip jako u Cannyho detektoru).
4. **Detekce pohybu** – rozdíl po sobě jdoucích snímků pozná ruku ve scéně; součet do uklidnění zamrzne a zobrazí „měřím…“.

Vyhodnocení: video se scénářem 10 mincí → odebrání → vrácení; graf surového a stabilizovaného součtu,
počet falešných změn součtu a doba reakce.

## Postup řešení

| # | Krok | Notebook |
|---|------|----------|
| 1 | Načtení snímku, HSV, histogramy | `01_nacteni_hsv_histogram.ipynb` |
| 2 | Předzpracování (filtrace šumu) | `02_predzpracovani.ipynb` |
| 3 | Segmentace (Otsu, adaptivní práh, morfologie) | `03_segmentace.ipynb` |
| 4 | Dotýkající se mince (watershed, Hough) | `04_watershed_hough.ipynb` |
| 5 | Příznaky a klasifikace | `05_priznaky_klasifikace.ipynb` |
| 6 | Evaluace, perceptron/MLP | `06_evaluace_mlp.ipynb` |
| 7 | Živý obraz z kamery, stabilní součet | `07_live_kamera.ipynb` |
| 8 | Vylepšení: CNN na výřezech mincí | `08_cnn.ipynb` |
| 9 | Volitelně: YOLO | `09_yolo.ipynb` |

## Struktura

```
notebooks/    jednotlivé kroky řešení
src/coins/    sdílené funkce (načítání, segmentace, příznaky, klasifikace)
data/raw/     vlastní fotky (není v gitu)
data/sample/  pár ukázkových fotek (v gitu)
results/      grafy a výsledky evaluace
DEVLOG.md     týdenní záznam práce
```

## Spuštění

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
nbstripout --install        # maže výstupy notebooků před commitem
jupyter lab
```
