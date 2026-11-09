# Zadání projektu: Počítání mincí na podložce

**Předmět:** Počítačové vidění, PEF MENDELU, ZS 2026/27
**Řešitelé:** Jakub Procházka (xproch40), Jan Kostrhun
**Datum:** 9. 11. 2026

## 1. Popis problému a cíl

Zákazník potřebuje systém, který z obrazu kamery umístěné nad podložkou automaticky rozpozná české oběžné mince (1, 2, 5, 10, 20 a 50 Kč), u každé zobrazí její hodnotu a v rohu obrazu průběžně ukazuje celkovou částku.

Systém pracuje v reálném čase nad videem. Součet musí být stabilní: nesmí poskakovat mezi snímky a po přidání nebo odebrání mince se má změnit do 1 s.

**Rozsah:** řízená scéna, tedy kamera v pevné výšce kolmo nad jednobarevnou matnou podložkou, rovnoměrné rozptýlené osvětlení a na podložce pouze české oběžné mince, volně vedle sebe nebo dotýkající se. Mimo rozsah jsou mince ležící přes sebe, bankovky, pamětní mince a libovolné pozadí. Vyřazování cizích předmětů (euro, prstýnek apod.) je volitelné rozšíření, pokud zbude čas.

**Akceptační kritéria (na testovacím datasetu):**

Klasifikátor ke každé minci vrací skóre příslušnosti ke každé třídě (pravděpodobnost z MLP/CNN, u klasického klasifikátoru vzdálenost v prostoru příznaků). Pro každou hodnotu mince sestavíme ROC křivku (jedna třída proti ostatním, změnou rozhodovacího prahu) a z ní určíme **EER (Equal Error Rate)**, tedy chybovost v bodě, kde je podíl falešně přijatých (FAR) roven podílu falešně odmítnutých (FRR).

- EER ≤ 5 % pro každou třídu mince, průměr přes třídy ≤ 3 %,
- AUC ≥ 0,98 pro každou třídu,
- přesně správný součet u ≥ 90 % snímků (kontrola celého systému),
- živý režim ≥ 10 snímků/s na běžném notebooku.

Vyhodnocení doplní matice záměn, ze které je vidět, které mince se pletou (očekáváme hlavně 1 Kč × 2 Kč a 5 Kč × 10 Kč).

## 2. Metody řešení

Řešení stavíme v Pythonu (OpenCV, NumPy, scikit-learn, PyTorch) v Jupyter noteboocích. Klasický řetězec zpracování doplníme o metody strojového učení.

1. **Snímání a kalibrace:** pevná expozice kamery, přepočet pixelů na milimetry podle známého rozměru podložky.
2. **Předzpracování:** převod do HSV a práce s jasovým kanálem V, mediánový filtr proti prachu na podložce (impulsní šum).
3. **Segmentace:** Otsuovo nebo adaptivní prahování, morfologické otevření a uzavření, oddělení dotýkajících se mincí pomocí distance transform a watershed. Pro srovnání Houghova transformace pro kružnice.
4. **Příznaky:** průměr v mm, průměrná sytost a odstín uvnitř mince (stříbrné 1–5 Kč, měděná 10 Kč, mosazná 20 Kč, bimetalová 50 Kč), kruhovitost (2 Kč je 11hranná, 20 Kč 13hranná).
5. **Klasifikace:** základní klasifikátor podle nejbližšího souseda v prostoru příznaků, poté perceptron/MLP. Výstupem je skóre pro každou třídu, ze kterého počítáme ROC a EER.
6. **Stabilní živý součet:** sledování mincí mezi snímky, hlasování o hodnotě přes historii mince, medián součtu v čase, hystereze při přidání nebo odebrání mince a pozastavení během pohybu ruky ve scéně.
7. **Vylepšení:** konvoluční neuronová síť nad výřezy mincí (s augmentací) a srovnání s klasickým řešením podle EER, potřeby dat a robustnosti.
8. **Volitelně, pokud zbude čas:** vyřazování cizích předmětů (předmět daleko od všech tříd = „není CZK mince“) a detektor YOLO.

## 3. Dostupné datasety

| Zdroj | Obsah | Využití |
|---|---|---|
| [ČNB – oběžné mince](https://www.cnb.cz/cs/bankovky-a-mince/mince/) | oficiální parametry: průměry 20,0 / 21,5 / 23,0 / 24,5 / 26,0 / 27,5 mm, materiál, tvar | referenční hodnoty pro klasifikaci podle velikosti |
| [Roboflow Universe](https://universe.roboflow.com/) | desítky veřejných datasetů mincí (eurové, polské, obecná třída „coin“), formát YOLO | pouze volitelné předtrénování detektoru, třídy CZK neobsahují |

Veřejný anotovaný dataset českých mincí jsme nenašli. Navíc hodnoty jasu a barvy závisí na konkrétní kameře, podložce a světle. Hlavní dataset proto nasnímáme sami.

## 4. Plán nasnímání vlastního datasetu

- **Sestava:** zapůjčená kamera na stojanu v pevné výšce, černá matná podložka, rozptýlené světlo. Totožná sestava pro všechny snímky.
- **Statické snímky (~200):**
  - každá hodnota mince samostatně, obě strany, různé natočení (≥ 20 výskytů na třídu),
  - mince od sebe a mince dotýkající se,
  - mix všech hodnot (5–15 mincí na snímek),
  - volitelně malá sada s cizími předměty (euro, prstýnek) pro rozšíření.
- **Videa (~10 × 30 s):** scénáře „10 mincí → odebrání jedné → vrácení“, přidávání mincí rukou, pohyb ruky v záběru.
- **Kontrola robustnosti:** malá sada se změněným osvětlením a výškou kamery.
- **Anotace:** CSV `soubor; počet mincí podle hodnot; součet; cizí předměty`, pro CNN navíc výřezy mincí s třídou.
- **Rozdělení:** 70 % trénovací, 15 % validační, 15 % testovací sada, vyvážené podle tříd. Testovací sada se neaugmentuje a nepoužívá se při ladění.

## 5. Odhad pracnosti a nacenění

| Etapa | Hodiny |
|---|---|
| Analýza, příprava prostředí a repozitáře | 6 |
| Nasnímání a anotace datasetu | 16 |
| Předzpracování a segmentace (vč. watershed/Hough) | 16 |
| Příznaky, kalibrace, klasifikace (pravidla, MLP) | 16 |
| Evaluace (ROC, EER, matice záměn) a ladění | 10 |
| Živý režim a stabilní součet | 16 |
| CNN nad výřezy mincí a srovnání | 16 |
| Dokumentace, prezentace, předání | 10 |
| Rezerva (cca 10 %) | 12 |
| **Celkem (2 řešitelé)** | **118** |

Při sazbě 600 Kč/h (junior vývojář) činí cena práce **70 800 Kč**. K tomu stojan a podložka cca 500 Kč, kamera je zapůjčena. Cena je orientační a nezahrnuje volitelná rozšíření: vyřazování cizích předmětů (+8 h, 4 800 Kč) a YOLO (+12 h, 7 200 Kč).

## 6. Výstupy

Git repozitář s notebooky jednotlivých kroků, knihovnou `src/coins`, ukázkovými daty a README; anotovaný testovací dataset; aplikace pro živý režim; vyhodnocení (ROC křivky a EER po třídách, matice záměn, správnost součtu, graf stabilizace součtu) a krátká technická zpráva.
