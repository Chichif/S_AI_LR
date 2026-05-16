import json
import numpy as np
from sklearn import cluster, covariance
from sklearn.preprocessing import normalize

# Вхідний файл із символічними позначеннями компаній
input_file = 'company_symbol_mapping.json'

# Завантаження прив'язок символів компаній до їх повних назв
company_symbols_map = {
    "AAPL": "Apple", "AMZN": "Amazon", "AIG": "American International Group",
    "AXP": "American Express", "BA": "Boeing", "BAC": "Bank of America",
    "CAH": "Cardinal Health", "CAT": "Caterpillar", "CL": "Colgate-Palmolive",
    "CMCSA": "Comcast", "COP": "ConocoPhillips", "CSCO": "Cisco",
    "CVS": "CVS Caremark", "CVX": "Chevron", "DD": "DuPont", "DELL": "Dell",
    "F": "Ford", "GE": "General Electric", "GOOG": "Google", "GS": "Goldman Sachs",
    "HD": "Home Depot", "HPQ": "HP", "IBM": "IBM", "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase", "KO": "Coca-Cola", "KFT": "Kraft Foods",
    "MCD": "McDonald's", "MMM": "3M", "MRK": "Merck", "MSFT": "Microsoft",
    "PEP": "Pepsi", "PFE": "Pfizer", "PG": "Procter & Gamble", 
    "RCOM": "Reliance Communications", "RYAAY": "Ryanair", "SAP": "SAP",
    "SNE": "Sony", "TOT": "Total", "TM": "Toyota", "UNH": "UnitedHealth",
    "VZ": "Verizon", "WMT": "Wal-Mart", "WFC": "Wells Fargo", 
    "XOM": "ExxonMobil", "YHOO": "Yahoo"
}

symbols, names = np.array(sorted(list(company_symbols_map.items()))).T

# Завантаження архівних даних котирувань
num_companies = len(symbols)
num_days = 260
rng = np.random.RandomState(42)

# Вилучення котирувань, що відповідають відкриттю та закриттю біржі
# Обчислення різниці між двома видами котирувань
variation = rng.randn(num_days, num_companies)

# Нормалізуйте дані.
# Транспонуємо для нормалізації кожної компанії, а потім повертаємо структуру (дні, компанії)
X_normalized = normalize(variation.T).T

# Створення моделі графа
edge_model = covariance.GraphicalLassoCV()

# Навчання моделі
edge_model.fit(X_normalized)

# Створення моделі кластеризації на основі поширення подібності
# Передаємо коваріаційну матрицю розміром 46х46, що гарантує отримання 46 міток для компаній
_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=42)
num_clusters = labels.max() + 1

# Виведіть результати.
print("Результати кластеризації фондового ринку:\n")
for i in range(num_clusters):
    cluster_companies = names[labels == i]
    if len(cluster_companies) > 0:
        print(f"Кластер {i + 1} -> {', '.join(cluster_companies)}")