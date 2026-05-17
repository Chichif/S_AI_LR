import datetime
import requests
import pandas as pd

class AirPollutionAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/air_pollution/history"

    def get_historical_data(self, lat, lon, start, end):
        params = {
            "lat": lat,
            "lon": lon,
            "start": start,
            "end": end,
            "appid": self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
            
        data = response.json()
        
        records = []
        for item in data.get("list", []):
            record = {
                "timestamp": datetime.datetime.fromtimestamp(item["dt"]).strftime('%Y-%m-%d %H:%M:%S'),
                "aqi": item["main"]["aqi"],
                "co": item["components"]["co"],
                "no": item["components"]["no"],
                "no2": item["components"]["no2"],
                "o3": item["components"]["o3"],
                "so2": item["components"]["so2"],
                "pm2_5": item["components"]["pm2_5"],
                "pm10": item["components"]["pm10"],
                "nh3": item["components"]["nh3"]
            }
            records.append(record)
            
        return pd.DataFrame(records)

def date_to_timestamp(date_str):
    dt = datetime.datetime.strptime(date_str, "%d-%m-%Y")
    return int(dt.timestamp())

def main():
    API_KEY = "53b64cd767987fdfd4ae26f0e648028c"
    
    locations = {
        "1": {"city": "Лондон", "lat": 51.5074, "lon": -0.1278},
        "2": {"city": "Варшава", "lat": 52.2297, "lon": 21.0122},
        "3": {"city": "Краків", "lat": 50.0647, "lon": 19.9450}
    }
    
    print("=== Моніторинг екологічних даних ===")
    print("1. Вибір попередньо встановленої локації")
    print("2. Ручне введення координат")
    choice = input("Ваш вибір (1/2): ").strip()
    
    if choice == "1":
        print("\nДоступні міста:")
        for k, v in locations.items():
            print(f"{k}. {v['city']} (lat: {v['lat']}, lon: {v['lon']})")
        loc_choice = input("Оберіть номер міста: ").strip()
        
        if loc_choice in locations:
            lat = locations[loc_choice]["lat"]
            lon = locations[loc_choice]["lon"]
            city_name = locations[loc_choice]["city"]
        else:
            print("Невірний вибір міста. Роботу завершено.")
            return
    elif choice == "2":
        try:
            lat = float(input("Введіть широту (lat від -90 до 90): "))
            lon = float(input("Введіть довготу (lon від -180 до 180): "))
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                print("Координати виходять за допустимі межі. Роботу завершено.")
                return
            city_name = "Задані координати"
        except ValueError:
            print("Некоректний формат чисел. Роботу завершено.")
            return
    else:
        print("Невірний вибір режиму. Роботу завершено.")
        return

    try:
        start_date_str = input("Введіть дату початку (DD-MM-YYYY): ").strip()
        end_date_str = input("Введіть дату кінця (DD-MM-YYYY): ").strip()
        
        start_ts = date_to_timestamp(start_date_str)
        end_ts = date_to_timestamp(end_date_str)
        
        if start_ts > end_ts:
            print("Помилка: Дата початку не може бути більшою за дату кінця.")
            return
    except ValueError:
        print("Помилка: Неправильний формат дати. Використовуйте DD-MM-YYYY.")
        return

    print(f"\nЗапит даних для локації: {city_name} ({lat}, {lon})...")
    
    try:
        api = AirPollutionAPI(API_KEY)
        df = api.get_historical_data(lat, lon, start_ts, end_ts)
        
        if df.empty:
            print("Дані за вказаний період відсутні або порожні.")
        else:
            print("\n=== Результати запиту (Перші 5 записів) ===")
            print(df.head().to_string(index=False))
            
            output_file = f"pollution_{city_name.lower().replace(' ', '_')}.csv"
            df.to_csv(output_file, index=False)
            print(f"\nВсі дані успішно збережено у файл: {output_file}")
            
    except Exception as e:
        print(f"\nВиникла помилка під час виконання запиту: {e}")

if __name__ == "__main__":
    main()