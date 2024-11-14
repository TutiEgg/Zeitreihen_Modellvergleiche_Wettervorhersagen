# Wetterdaten-Vorhersage mit Deep Learning und Zeitreihenanalyse

Dieses Projekt vergleicht verschiedene Modelle zur Vorhersage von Wetterdaten-Zeitreihen, darunter klassische statistische Methoden und moderne Deep-Learning-Ansätze, um die Vorhersagegenauigkeit und Effizienz der Methoden zu untersuchen.

## Inhaltsverzeichnis
1. [Einleitung](#einleitung)
2. [Verwendete Modelle](#verwendete-modelle)
3. [Datensatz und Datenvorverarbeitung](#datensatz-und-datenvorverarbeitung)
4. [Modelle und Methodik](#modelle-und-methodik)
5. [Ergebnisse](#ergebnisse)
6. [Diskussion](#diskussion)
7. [Fazit](#fazit)

## Einleitung

![Einführung in die Wettervorhersage-Methoden](path/to/image1.png)

Wettervorhersagen basieren auf mathematischen Modellen der Atmosphäre und nutzen zunehmend Deep-Learning-Ansätze. Dieser Vergleich zielt darauf ab, verschiedene Modellarchitekturen wie LSTM und Transformer mit traditionellen Methoden wie SARIMA zu evaluieren.

## Verwendete Modelle

Die betrachteten Modelle umfassen:

- **SARIMA**: Ein klassisches Modell für stationäre Zeitreihen, das saisonale Effekte berücksichtigt.
- **LSTM**: Ein rekurrentes neuronales Netzwerk, das langfristige Abhängigkeiten lernt und ideal für sequentielle Daten ist.
- **Transformer**: Modelliert Abhängigkeiten effizient mit einem Attention-Mechanismus, was sich besonders für längere Sequenzen eignet.

## Datensatz und Datenvorverarbeitung

![Datensatz-Visualisierung](path/to/image2.png)

Der verwendete Datensatz, bereitgestellt vom Max-Planck-Institut, umfasst meteorologische Daten von 2009 bis 2016. Die Daten enthalten 14 Variablen, darunter Temperatur, Luftdruck und Windgeschwindigkeit, mit einer ursprünglichen Abtastrate von 10 Minuten. Der Datensatz wurde in stündliche Intervalle umgewandelt, und fehlerhafte Werte wurden bereinigt.

### Datenvorverarbeitungsschritte:
1. **Korrektur fehlerhafter Werte**: Werte wie -9999 bei Windgeschwindigkeit wurden auf 0.0 gesetzt.
2. **Umwandlung der Winddaten**: Windrichtung und -geschwindigkeit wurden in Vektoren umgewandelt, um Kontinuität zu gewährleisten.
3. **Normalisierung**: Die Daten wurden auf einen einheitlichen Wertebereich skaliert.

## Modelle und Methodik

### SARIMA-Modell

![SARIMA Modell](path/to/image3.png)

Das SARIMA-Modell nutzt saisonale und nicht-saisonale Parameter zur Modellierung der Zeitreihe. Der Augmented Dickey-Fuller-Test wurde zur Sicherstellung der Stationarität der Daten angewandt. Eine Vielzahl an Modellkonfigurationen wurde getestet, um die besten Parameter mittels Akaike-Informationskriterium (AIC) zu identifizieren.

### LSTM-Modell

![LSTM Modell](path/to/image4.png)

Das LSTM-Modell besteht aus einer LSTM-Schicht und einer voll verbundenen Schicht für die Ausgabe. Die Eingabedaten wurden mit einem „WindowGenerator“ vorstrukturiert, der Eingabefenster für historische Daten und die Zielwerte für Vorhersagen erstellt.

### Transformer-Modell

![Transformer Architektur](path/to/image5.png)

Das Transformer-Modell nutzt eine Encoder-Decoder-Architektur mit einer Maskierungsfunktion, die sicherstellt, dass nur vergangene Daten zur Vorhersage herangezogen werden. Die Daten werden durch ein Sliding-Window-Verfahren strukturiert und in 16-dimensionale Embeddings überführt.

## Ergebnisse

Die Ergebnisse zeigen deutliche Unterschiede in der Genauigkeit und der Effizienz der Modelle:

- **SARIMA**: Liefert schnelle und zuverlässige Vorhersagen für stationäre Zeitreihen.
- **LSTM**: Geeignet für größere, nicht-stationäre Datensätze; benötigt jedoch mehr Zeit für das Training.
- **Transformer**: Zeigt besonders bei langen Zeitreihen hohe Genauigkeit, profitiert von Attention-Mechanismen für komplexe Muster.

## Diskussion

- **Feature-Auswahl**: Zeitmerkmale wie "Tag des Monats" verbessern die Modellgenauigkeit.
- **Overfitting**: Transformer neigen zu Overfitting bei kleineren Datensätzen.
- **Frequenzabhängigkeit**: Unterschiedliche Datenfrequenzen beeinflussen die Vorhersagegenauigkeit.

## Fazit

Die Ergebnisse zeigen, dass die Wahl des Modells von der Art der Zeitreihe abhängt:

- **SARIMA**: Effizient für schnelle Vorhersagen bei stationären Daten.
- **LSTM**: Eignet sich für nicht-stationäre Zeitreihen und zeigt bei großen Datensätzen gute Leistungen.
- **Transformer**: Ideal für komplexe, saisonale Muster in großen Datenmengen.

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz verfügbar. Weitere Details siehe [LICENSE](LICENSE).



# HOW TO

## SARIMA

Es wurden die wichtigsten Notebooks für die SARIMA-Modelle im "SARIMA_CODE" - Ordner hinterlegt, um einen Vergleich zwischen den verschiedenen Kombinationen der Daten-Modelle zu veranschaulichen.

Um andere Datensätze zu testen, müssen Sie in der zweiten Zelle den dementsprechend anderen Datensatz importieren. Es ist bei individuellen Veränderungsversuchen den Notebook: "Eval_SARIMA_weather_6h_d0_D0_s4_exog" für Datensätze ohne Differenzierungen und den Notebook: "Eval_SARIMA_weather_6h_d1_D1_s4_exog"  für Datensätze mit d=1 und D=1 zu benutzen

## LSTM
Für die LSTM Notebooks muss wie bei den SARIMAS der gewünschten Datensatz importiert werden und folgende Einstellungen vorgenommen werde:
- input_width = z.B 24
- label_width  = z.B 24
- offset = z.B 24
- predict_label = z.B ["T_(degC)"]

## Transformer
Bei den Transformermodellen muss lediglich die main mit den passenden Pfaden zu den jeweiligen Datein verändert werden.
Hyperparameter sind in der config im data-Ordner einstellbar. 
Folgende Einstellungen können vorgenommen werden:
- features: Merkmale die im Training für das "target"-Vorhersehen benutzt werden sollen.
- target: Label welche als output ausgegeben werden sollen (Vorhersage_output)
- group_by_key: Gruppierung ["day_of_month","day_of_year"]
- lag_features: Geben keine merkbaren Verbesserungen im Training
- epochs: Epochen
- bactch_size: Anzahl an batches. Mehr Batches weniger Schritte pro Datengruppe (=schlechtere Resultate)
- horizon_size: Anzahl zukünftiger Werte
- history_size: Anzahl vergangener Werte
- channels: Channel anzahl
- lr: Learning rate
- dropout: dropout chance

## Create_Data
In dieser Datei findet die vorverarbeitung des Datensatzes statt.

## Environment
Es wurde mit Python 3.10 gearbeitet.
                                                                                                                                                            
