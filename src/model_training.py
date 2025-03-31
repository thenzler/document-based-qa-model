import random
import json
from datetime import datetime

class ChurnModel:
    """
    Klasse für Churn-Vorhersagemodell
    """
    
    def __init__(self):
        """
        Initialisiert das Churn-Vorhersagemodell
        """
        self.model_trained = False
        self.training_date = None
    
    def train(self, data=None):
        """
        Trainiert das Modell mit Trainingsdaten
        
        Args:
            data (list): Trainingsdaten als Liste von Kundendaten-Dicts
        
        Returns:
            dict: Trainingsergebnis
        """
        # Für eine Demo-Version simulieren wir das Training
        
        # Simuliere Trainingszeit
        import time
        time.sleep(2)
        
        self.model_trained = True
        self.training_date = datetime.now()
        
        return {
            "success": True,
            "accuracy": random.uniform(0.75, 0.92),
            "f1_score": random.uniform(0.70, 0.90),
            "training_time": random.uniform(2.5, 10.0),
            "training_date": self.training_date.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def predict(self, data):
        """
        Führt Churn-Vorhersagen für Kundendaten durch
        
        Args:
            data (list): Liste von Kundendaten-Dicts
        
        Returns:
            dict: Vorhersageergebnisse
        """
        # Prüfe, ob Daten vorhanden sind
        if not data:
            # Simuliere Beispieldaten
            data = self._get_mock_customer_data(10)
        
        # Für eine Demo-Version simulieren wir die Vorhersage
        predictions = []
        risk_categories = ["niedrig", "mittel", "hoch", "sehr hoch"]
        risk_factors = [
            "Geringe Produktnutzung", 
            "Zahlungsverzögerungen", 
            "Keine Reaktion auf Marketing-Kampagnen",
            "Häufige Support-Anfragen",
            "Reduzierte Nutzungsintensität",
            "Keine Interaktion auf sozialen Medien",
            "Lange Inaktivität"
        ]
        
        actions = [
            "Persönlichen Kontakt aufnehmen", 
            "Spezialangebot unterbreiten", 
            "Feedback einholen",
            "Produkt-Schulung anbieten",
            "Proaktiven Support bieten",
            "Rabatt für Verlängerung anbieten",
            "Check-in-Gespräch vereinbaren"
        ]
        
        # Simuliere Vorhersagen für jeden Kunden
        for customer in data:
            # Simuliere Churn-Wahrscheinlichkeit
            churn_probability = random.uniform(0.1, 0.9)
            
            # Bestimme Risikokategorie
            risk_index = min(int(churn_probability * 4), 3)
            risk_category = risk_categories[risk_index]
            
            # Wähle zufällige Risikofaktoren
            num_factors = random.randint(1, 3)
            selected_factors = random.sample(risk_factors, num_factors)
            
            # Wähle zugehörige Handlungsempfehlungen
            num_actions = random.randint(1, 3)
            selected_actions = random.sample(actions, num_actions)
            
            # Erstelle Vorhersage-Objekt
            prediction = {
                "customer_id": customer.get("customer_id", str(random.randint(1000, 9999))),
                "name": customer.get("name", f"Kunde {random.randint(1000, 9999)}"),
                "churn_probability": churn_probability,
                "risk_category": risk_category,
                "risk_factors": selected_factors,
                "recommended_actions": selected_actions,
                "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            predictions.append(prediction)
        
        # Erstelle Zusammenfassung
        summary = {
            "total_customers": len(predictions),
            "risk_distribution": {
                "niedrig": len([p for p in predictions if p["risk_category"] == "niedrig"]),
                "mittel": len([p for p in predictions if p["risk_category"] == "mittel"]),
                "hoch": len([p for p in predictions if p["risk_category"] == "hoch"]),
                "sehr hoch": len([p for p in predictions if p["risk_category"] == "sehr hoch"])
            },
            "avg_churn_probability": sum(p["churn_probability"] for p in predictions) / len(predictions)
        }
        
        return {
            "predictions": predictions,
            "summary": summary
        }
    
    def _get_mock_customer_data(self, num_customers=10):
        """
        Erstellt simulierte Kundendaten für Demozwecke
        
        Args:
            num_customers (int): Anzahl der zu erstellenden Kundendatensätze
            
        Returns:
            list: Liste mit simulierten Kundendaten
        """
        customer_data = []
        
        first_names = ["Anna", "Max", "Julia", "Thomas", "Sarah", "Michael", "Laura", "David", "Maria", "Felix"]
        last_names = ["Müller", "Schmidt", "Weber", "Schneider", "Fischer", "Meyer", "Wagner", "Becker", "Hoffmann", "Koch"]
        
        products = ["Premium-Abo", "Standard-Abo", "Basic-Abo"]
        
        for i in range(num_customers):
            # Erstelle einen simulierten Kundendatensatz
            customer = {
                "customer_id": f"CID{random.randint(10000, 99999)}",
                "name": f"{random.choice(first_names)} {random.choice(last_names)}",
                "email": f"kunde{i+1}@example.com",
                "product": random.choice(products),
                "subscription_months": random.randint(1, 36),
                "monthly_charges": round(random.uniform(20, 100), 2),
                "total_charges": round(random.uniform(100, 3000), 2),
                "last_login_days": random.randint(0, 60),
                "support_tickets": random.randint(0, 10),
                "payment_delay_count": random.randint(0, 5)
            }
            
            customer_data.append(customer)
        
        return customer_data
