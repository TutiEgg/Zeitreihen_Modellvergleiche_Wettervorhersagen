# SARIMA

Es wurden die wichtigsten Notebooks für die SARIMA-Modelle im "SARIMA_CODE" - Ordner hinterlegt, um einen Vergleich zwischen den verschiedenen Kombinationen der Daten-Modelle zu veranschaulichen.

Um andere Datensätze zu testen, müssen Sie in der zweiten Zelle den dementsprechend anderen Datensatz importieren. Es ist bei individuellen Veränderungsversuchen den Notebook: "Eval_SARIMA_weather_6h_d0_D0_s4_exog" für Datensätze ohne Differenzierungen und den Notebook: "Eval_SARIMA_weather_6h_d1_D1_s4_exog"  für Datensätze mit d=1 und D=1 zu benutzen

# LSTM
Für die LSTM Notebooks muss wie bei den SARIMAS der gewünschten Datensatz importiert werden und folgende Einstellungen vorgenommen werde:
- input_width = z.B 24
- label_width  = z.B 24
- offset = z.B 24
- predict_label = z.B ["T_(degC)"]

# Transformer
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

# Create_Data
In dieser Datei findet die vorverarbeitung des Datensatzes statt.

# Environment
Es wurde mit Python 3.10 gearbeitet.
                                                                                                                                                            