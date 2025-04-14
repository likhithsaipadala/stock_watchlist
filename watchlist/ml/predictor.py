def predict_next_price(ticker):
    # Dummy logic — replace with real ML model later
    import random
    current_price = random.uniform(1000, 3000)
    predicted_price = current_price * random.uniform(0.95, 1.05)
    return round(predicted_price, 2)
