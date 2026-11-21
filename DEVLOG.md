# DEVLOG

Krátký záznam po každém týdnu: co jsme udělali, co nefungovalo a proč.

## Týden 1 (2.–8. 11. 2026)

- Založen repozitář, struktura projektu, README, requirements.
- Nafocené první snímky mobilem na černé podložce
- Notebook `01_nacteni_hsv_histogram`: histogram jasu je na černé podložce bimodální → v kroku 03 zkusit Otsu na kanálu V.
- Sytost S odliší stříbrné (1, 2, 5 Kč, S ≈ 65–85) od zlatých mincí (20, 50 Kč, S ≈ 140); na tmavém pozadí je ale S nespolehlivá.
- Notebook `02_predzpracovani`: porovnání filtrů na kanálu V. Prach na podložce se chová jako impulsní šum

## Konzultace zadání (9. 11. 2026)

- Zadání schváleno. Vyřazování cizích předmětů jen jako volitelné rozšíření, pokud zbude čas.
- Akceptační kritéria přepsána na ROC křivky a EER pro každou hodnotu mince (+ matice záměn).
- Nové fotky 3–5.jpg s dotýkajícími se mincemi (převedeny z HEIC, originály v `data/raw/`).
- Notebook `04_watershed_hough`: distance transform → lokální maxima jako markery → watershed. Výsledek 11/11, 2/2, 13/13, 4/4, 13/13.
- Hough je citlivý na `param2` (5.jpg: 31/21/14 kruhů místo 13) a potřebuje rozsah poloměrů → hlavní metoda watershed.
- Oprava kroku 03: uzavření 1× a vyplňují se jen malé díry, jinak se slily mince natěsno.
- Fotky jsou focené z různé výšky (50 Kč má r = 58–88 px) → pro krok 05 nutná kalibrace px → mm.


## Týden 2 (9.–15. 11. 2026)

- Nafocen vlastní dataset (14. 11.): iPhone 1,5×, 30 cm nad podložkou, pravítko v záběru. 88 fotek ve složkách
  `samostatne`, `mix`, `dotyk`, `test`, `robust`; v gitu zmenšené na 1428 px (`data/dataset/`), originály na Drive.
- Anotace: `anotace.csv` (počty podle hodnot, součet) a `anotace_mince.csv` (poloha a hodnota každé z 683 mincí).
- Notebook `05_priznaky_klasifikace`:
  - Stará segmentace na šedé podložce vynechávala tmavé lesklé 1 a 2 Kč (460/476). Nově odečtení pozadí
    (mediánový filtr s velkým jádrem), maska pravítka podle červené barvy a kruhovitost na konvexním obalu → 476/476.
  - Kalibrace 4,876 px/mm podle mincí známého průměru (pravítko je v záběru oříznuté). Průměr = menší osa proložené
    elipsy; opsaná kružnice u dotýkajících se mincí vycházela o 0,5 mm větší.
  - Klasifikátor podle nejbližšího souseda se skóre pro každou třídu (−vzdálenost). Na validaci vybrány příznaky
    průměr + sytost uprostřed; odstín výsledek zhoršil (u stříbrných mincí šum).
  - Test: 99,4 % mincí správně, přesný součet na 15/16 snímků (93,8 %). Chyba = dvě slité mince v detekci.
  - Boční světlo (`robust/`): jen 74 % mincí a 1/6 součtů – sytost i okraj masky se světlem mění. Hlavní slabina.
- Notebook `06_evaluace_mlp`: ROC, EER a AUC pro každou hodnotu (jedna proti ostatním), srovnání kNN, perceptronu a MLP.
  - Validace (`dotyk/`) je nasycená: kNN i všechny MLP 100 %. Vybrána nejmenší síť (8 neuronů, α = 0,01).
  - Perceptron nestačí (93 %, AUC 0,82): hodnoty uprostřed osy průměru nejdou oddělit jednou přímkou.
  - Test: kNN 99,4 % mincí, součty 93,8 %, ale EER 1 Kč 9,3 % a 2 Kč 7,1 % (> 5 %). MLP splní EER i AUC
    (průměr 1,4 %, max 4,1 %, AUC ≥ 0,996), ale součty jen 81 %. Žádný model nesplní všechna kritéria najednou.
  - Příčina: v testu jsou zašlé „růžové“ 1 Kč (sytost 67–84, v trénovacích datech max 59). Validace z týchž kusů
    mincí to neodhalila → dofotit více kusů 1 a 2 Kč, validaci dělat z jiných kusů než trénink.

## Týden 3 (16.–22. 11. 2026)

- Vyučující povolil iPhone jako kameru (Continuity Camera).
- Notebook `07_live_kamera` a skript `live.py`: obraz z kamery → detekce → kNN → hodnota u mince a součet v rohu.
  Snímek se přeškáluje na měřítko datasetu (4,876 px/mm), kalibrace klávesou k z mincí jedné hodnoty.
- Příznaky jednou za snímek a jen ve výřezu kolem mince (`frame_features`): 200 ms → 3 ms. Celý snímek ~80 ms (~12 fps).
- Stabilní součet (`src/coins/stabilize.py`): sledování mincí, hlasování o hodnotě, hystereze 3/3 snímky, medián z 5,
  zmrazení při pohybu. Simulace (mince → ruka → odebraná 50 Kč): surový součet se změní 15×, stabilní 2×, reakce 0,8 s.
- Další krok: ověřit na reálném videu z iPhonu (`--log`), případně přeučit na snímcích z videa.

