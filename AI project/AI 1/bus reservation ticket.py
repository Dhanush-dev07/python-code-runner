"""
AI-Powered Bus Ticket Reservation System
-----------------------------------------
A console-based mini project demonstrating:
  1. Core reservation logic (search, book, cancel, view tickets)
  2. A simple AI demand-forecasting model (predicts how full a trip
     will get, using historical booking counts per route)
  3. A rule-based NLP "chat assistant" that answers booking queries
     in plain language

Data is persisted to a local JSON file (bus_data.json) so bookings
survive between runs. No external libraries required.

Run:  python bus_reservation.py
"""

import json
import os
import re
from datetime import datetime

DATA_FILE = "bus_data.json"


# ----------------------------------------------------------------------
# 1. DATA MODEL
# ----------------------------------------------------------------------

class Bus:
    """Represents a single bus trip on a route."""

    def __init__(self, bus_id, source, destination, depart_time,
                 total_seats, base_fare, booked_seats=None, history=None):
        self.bus_id = bus_id
        self.source = source
        self.destination = destination
        self.depart_time = depart_time
        self.total_seats = total_seats
        self.base_fare = base_fare
        self.booked_seats = set(booked_seats or [])
        # history: list of past occupancy % (%) used for demand forecasting
        self.history = history or []

    @property
    def seats_left(self):
        return self.total_seats - len(self.booked_seats)

    @property
    def occupancy_pct(self):
        return round(len(self.booked_seats) / self.total_seats * 100, 1)

    def to_dict(self):
        return {
            "bus_id": self.bus_id,
            "source": self.source,
            "destination": self.destination,
            "depart_time": self.depart_time,
            "total_seats": self.total_seats,
            "base_fare": self.base_fare,
            "booked_seats": list(self.booked_seats),
            "history": self.history,
        }

    @staticmethod
    def from_dict(d):
        return Bus(d["bus_id"], d["source"], d["destination"], d["depart_time"],
                    d["total_seats"], d["base_fare"], d["booked_seats"], d["history"])


class Booking:
    """Represents a confirmed ticket."""

    def __init__(self, pnr, bus_id, passenger, seat_no, fare_paid, timestamp=None):
        self.pnr = pnr
        self.bus_id = bus_id
        self.passenger = passenger
        self.seat_no = seat_no
        self.fare_paid = fare_paid
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d):
        return Booking(**d)


# ----------------------------------------------------------------------
# 2. AI LAYER — demand forecasting + fare suggestion + chat assistant
# ----------------------------------------------------------------------

class DemandForecaster:
    """
    Very lightweight 'AI' model: predicts expected occupancy for a bus
    using a weighted moving average of its past occupancy history, then
    recommends a fare within a bounded dynamic-pricing band.

    (In the full system this would be a trained regression / time-series
    model over historical bookings; here it is a transparent statistical
    stand-in so the logic is easy to follow and extend.)
    """

    @staticmethod
    def predict_occupancy(bus: Bus) -> float:
        if not bus.history:
            return bus.occupancy_pct  # no history yet -> use current state
        weights = list(range(1, len(bus.history) + 1))  # recent trips weigh more
        weighted_sum = sum(h * w for h, w in zip(bus.history, weights))
        return round(weighted_sum / sum(weights), 1)

    @staticmethod
    def suggest_fare(bus: Bus) -> int:
        predicted = DemandForecaster.predict_occupancy(bus)
        fare = bus.base_fare
        if predicted >= 80:
            fare = round(bus.base_fare * 1.15)   # high demand -> +15%
        elif predicted <= 30:
            fare = round(bus.base_fare * 0.90)   # low demand -> -10%
        return fare

    @staticmethod
    def confidence_label(bus: Bus) -> str:
        n = len(bus.history)
        if n >= 5:
            return "High"
        elif n >= 2:
            return "Medium"
        return "Low"


class ChatAssistant:
    """
    Rule-based NLP assistant: extracts source/destination keywords and
    intent (cheapest / earliest / seats-left) from a free-text query and
    returns matching trips. Intentionally simple (keyword + intent rules)
    rather than a trained classifier, to keep the demo dependency-free.
    """

    INTENT_PATTERNS = {
        "cheapest": r"\bcheap(est)?\b|\blowest fare\b",
        "earliest": r"\bearl(y|iest)\b|\bfirst\b|\bmorning\b",
        "seats": r"\bseats?\b|\bavailab",
    }

    def __init__(self, system):
        self.system = system

    def _extract_route(self, text):
        text = text.lower()
        match = re.search(r"from\s+([a-z]+)\s+to\s+([a-z]+)", text)
        if match:
            return match.group(1).title(), match.group(2).title()
        # fallback: "chennai to madurai"
        match = re.search(r"([a-z]+)\s+to\s+([a-z]+)", text)
        if match:
            return match.group(1).title(), match.group(2).title()
        return None, None

    def _extract_intent(self, text):
        text = text.lower()
        for intent, pattern in self.INTENT_PATTERNS.items():
            if re.search(pattern, text):
                return intent
        return "default"

    def respond(self, query):
        source, destination = self._extract_route(query)
        if not source or not destination:
            return ("I couldn't find a route in that message. Try: "
                    "\"buses from Chennai to Madurai\"")

        trips = self.system.search_buses(source, destination)
        if not trips:
            return f"No buses found from {source} to {destination}."

        intent = self._extract_intent(query)
        if intent == "cheapest":
            trips.sort(key=lambda b: DemandForecaster.suggest_fare(b))
        elif intent == "earliest":
            trips.sort(key=lambda b: b.depart_time)
        elif intent == "seats":
            trips.sort(key=lambda b: -b.seats_left)

        best = trips[0]
        fare = DemandForecaster.suggest_fare(best)
        return (f"Best match: Bus {best.bus_id} departs {best.depart_time}, "
                f"fare \u20b9{fare}, {best.seats_left} seats left "
                f"(demand confidence: {DemandForecaster.confidence_label(best)}).")


# ----------------------------------------------------------------------
# 3. RESERVATION SYSTEM — orchestrates buses, bookings, persistence
# ----------------------------------------------------------------------

class ReservationSystem:
    def __init__(self):
        self.buses = {}      # bus_id -> Bus
        self.bookings = {}   # pnr -> Booking
        self._pnr_counter = 1000
        self.chat = ChatAssistant(self)
        self.load()
        if not self.buses:
            self._seed_demo_data()

    # ---- persistence ----
    def save(self):
        data = {
            "buses": [b.to_dict() for b in self.buses.values()],
            "bookings": [bk.to_dict() for bk in self.bookings.values()],
            "pnr_counter": self._pnr_counter,
        }
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        if not os.path.exists(DATA_FILE):
            return
        with open(DATA_FILE) as f:
            data = json.load(f)
        for bd in data.get("buses", []):
            bus = Bus.from_dict(bd)
            self.buses[bus.bus_id] = bus
        for bkd in data.get("bookings", []):
            bk = Booking.from_dict(bkd)
            self.bookings[bk.pnr] = bk
        self._pnr_counter = data.get("pnr_counter", 1000)

    def _seed_demo_data(self):
        demo = [
            Bus("BT101", "Chennai", "Madurai", "21:40", 40, 650, history=[62, 70, 88, 91]),
            Bus("BT102", "Chennai", "Madurai", "07:30", 40, 620, history=[40, 35, 28]),
            Bus("BT103", "Chennai", "Trichy", "19:15", 36, 480, history=[55, 60]),
            Bus("BT104", "Coimbatore", "Chennai", "22:00", 45, 720, history=[80, 85, 95, 90, 88]),
        ]
        for b in demo:
            self.buses[b.bus_id] = b
        self.save()

    # ---- core operations ----
    def search_buses(self, source, destination):
        return [b for b in self.buses.values()
                if b.source.lower() == source.lower()
                and b.destination.lower() == destination.lower()]

    def book_ticket(self, bus_id, passenger_name):
        bus = self.buses.get(bus_id)
        if not bus:
            return None, "Bus not found."
        if bus.seats_left <= 0:
            return None, "No seats available on this bus."

        seat_no = min(set(range(1, bus.total_seats + 1)) - bus.booked_seats)
        bus.booked_seats.add(seat_no)

        fare = DemandForecaster.suggest_fare(bus)
        self._pnr_counter += 1
        pnr = f"PNR{self._pnr_counter}"
        booking = Booking(pnr, bus_id, passenger_name, seat_no, fare)
        self.bookings[pnr] = booking

        self.save()
        return booking, None

    def cancel_ticket(self, pnr):
        booking = self.bookings.get(pnr)
        if not booking:
            return False, "PNR not found."
        bus = self.buses.get(booking.bus_id)
        if bus and booking.seat_no in bus.booked_seats:
            bus.booked_seats.remove(booking.seat_no)
        del self.bookings[pnr]
        self.save()
        return True, "Ticket cancelled."

    def bookings_for(self, passenger_name):
        return [bk for bk in self.bookings.values()
                if bk.passenger.lower() == passenger_name.lower()]


# ----------------------------------------------------------------------
# 4. CONSOLE MENU
# ----------------------------------------------------------------------

def print_bus_row(bus):
    fare = DemandForecaster.suggest_fare(bus)
    predicted = DemandForecaster.predict_occupancy(bus)
    print(f"  [{bus.bus_id}] {bus.source} -> {bus.destination}  "
          f"dep {bus.depart_time}  seats left {bus.seats_left}/{bus.total_seats}  "
          f"fare \u20b9{fare}  predicted demand {predicted}% "
          f"(confidence: {DemandForecaster.confidence_label(bus)})")


def main():
    system = ReservationSystem()

    menu = """
========== AI Bus Ticket Reservation ==========
1. Search buses
2. Book a ticket
3. Cancel a ticket
4. View my bookings
5. Ask the chat assistant
6. Exit
================================================
"""
    while True:
        print(menu)
        choice = input("Choose an option (1-6): ").strip()

        if choice == "1":
            source = input("From: ").strip()
            destination = input("To: ").strip()
            results = system.search_buses(source, destination)
            if not results:
                print("No buses found for that route.")
            else:
                print(f"\nBuses from {source} to {destination}:")
                for bus in results:
                    print_bus_row(bus)

        elif choice == "2":
            bus_id = input("Bus ID (see search results): ").strip().upper()
            name = input("Passenger name: ").strip()
            booking, error = system.book_ticket(bus_id, name)
            if error:
                print(f"Booking failed: {error}")
            else:
                print(f"\nBooked! PNR: {booking.pnr}  Seat: {booking.seat_no}  "
                      f"Fare: \u20b9{booking.fare_paid}")

        elif choice == "3":
            pnr = input("Enter PNR to cancel: ").strip().upper()
            ok, msg = system.cancel_ticket(pnr)
            print(msg)

        elif choice == "4":
            name = input("Passenger name: ").strip()
            results = system.bookings_for(name)
            if not results:
                print("No bookings found.")
            else:
                for bk in results:
                    bus = system.buses.get(bk.bus_id)
                    route = f"{bus.source} -> {bus.destination}" if bus else "?"
                    print(f"  {bk.pnr}  {route}  seat {bk.seat_no}  "
                          f"fare \u20b9{bk.fare_paid}  booked {bk.timestamp}")

        elif choice == "5":
            query = input('Ask something (e.g. "cheapest bus from Chennai to Madurai"): ')
            print(system.chat.respond(query))

        elif choice == "6":
            print("Thank you for using the AI Bus Reservation System. Safe travels!")
            break

        else:
            print("Invalid option, please choose 1-6.")


if __name__ == "__main__":
    main()
